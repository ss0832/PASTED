/**
 * pasted._ext._maxent_core  (v0.1.15)
 * =====================================
 * Two exported functions:
 *
 *   angular_repulsion_gradient(pts, cutoff)
 *       -> grad: ndarray(n, 3)
 *       Unchanged from v0.1.14.
 *
 *   place_maxent_cpp(pts, radii, cov_scale, region_radius, ang_cutoff,
 *                    maxent_steps, trust_radius, seed)
 *       -> pts_out: ndarray(n, 3)          [NEW in v0.1.15]
 *
 *       Runs the entire maxent gradient-descent loop in C++:
 *         for each step:
 *           1. Evaluate angular repulsion energy U and gradient g (L-BFGS)
 *           2. Compute L-BFGS descent direction d (m=7, Armijo backtracking)
 *           3. Per-atom trust-radius clip: rescale step so max atom disp <= trust_radius
 *           4. Step: x += d  (d is already a descent direction)
 *           5. Soft restoring force (atoms outside region_radius pulled back)
 *           6. Centre-of-mass pinning
 *           7. Steric-clash relaxation (L-BFGS penalty, identical to _relax_core)
 *
 * Dependencies: C++17 stdlib + pybind11.  No Eigen, no OpenMP.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <random>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace py = pybind11;
using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

static constexpr int    CELL_LIST_THRESHOLD = 32;
static constexpr double EPS                 = 1e-6;
static constexpr int    LBFGS_M             = 7;
static constexpr double ARMIJO_C1           = 1e-4;
static constexpr int    MAX_LS_STEPS        = 50;
static constexpr int    RELAX_CELL_THR      = 64;

// ===========================================================================
// Vec
// ===========================================================================
struct Vec {
    std::vector<double> d;
    Vec() = default;
    explicit Vec(int n, double v = 0.0) : d(std::size_t(n), v) {}
    int    size()  const noexcept { return int(d.size()); }
    double&       operator[](int i)       noexcept { return d[i]; }
    double        operator[](int i) const noexcept { return d[i]; }
    double*       data()       noexcept { return d.data(); }
    const double* data() const noexcept { return d.data(); }
    void zero() noexcept { std::fill(d.begin(), d.end(), 0.0); }
    double dot(const Vec& o) const noexcept {
        double s = 0; for (int i = 0; i < size(); ++i) s += d[i]*o.d[i]; return s;
    }
    double norm2() const noexcept { return dot(*this); }
    double norm()  const noexcept { return std::sqrt(norm2()); }
    void add_scaled(double a, const Vec& x) noexcept {
        for (int i = 0; i < size(); ++i) d[i] += a*x.d[i];
    }
    void assign_sum(const Vec& a, double alpha, const Vec& b) noexcept {
        for (int i = 0; i < size(); ++i) d[i] = a.d[i] + alpha*b.d[i];
    }
    void assign_scaled(double s, const Vec& x) noexcept {
        for (int i = 0; i < size(); ++i) d[i] = s*x.d[i];
    }
    void assign_diff(const Vec& x, const Vec& y) noexcept {
        for (int i = 0; i < size(); ++i) d[i] = x.d[i]-y.d[i];
    }
    void copy_from(const Vec& o) { d = o.d; }
    void copy_from(const double* src, int n) { d.assign(src, src+std::size_t(n)); }
    void copy_to(double* dst) const noexcept { std::copy(d.begin(), d.end(), dst); }
};

// ===========================================================================
// L-BFGS (m=7, Armijo backtracking)
// ===========================================================================
static std::pair<double,bool> lbfgs_minimize(
    Vec& x,
    std::function<double(const Vec&, Vec&)> eval,
    int max_iter, double etol = 1e-12)
{
    const int dim = x.size(), m = LBFGS_M;
    std::vector<Vec> s(m,Vec(dim)), y(m,Vec(dim));
    std::vector<double> rho(m,0), alp(m,0);
    int ptr=0, cnt=0;
    Vec g(dim),gn(dim),q(dim),r(dim),dd(dim),xt(dim);

    double E = eval(x,g);
    if (E<=etol) return {E,true};

    auto slot=[&](int i){return (ptr+m-1-i)%m;};

    for (int it=0; it<max_iter; ++it) {
        q.copy_from(g);
        for (int i=0;i<cnt;++i){int k=slot(i);alp[i]=rho[k]*s[k].dot(q);q.add_scaled(-alp[i],y[k]);}
        if (cnt>0){int k=slot(0);double sy=s[k].dot(y[k]),yy=y[k].norm2();
            r.assign_scaled(yy>1e-20?sy/yy:1.0,q);}
        else{double gn2=g.norm();if(gn2>1e-20)r.assign_scaled(1.0/gn2,q);else r.copy_from(q);}
        for (int i=cnt-1;i>=0;--i){int k=slot(i);double b=rho[k]*y[k].dot(r);r.add_scaled(alp[i]-b,s[k]);}
        dd.assign_scaled(-1.0,r);

        double dg0=dd.dot(g);
        if (dg0>=-1e-14*dd.norm()*g.norm()){dd.assign_scaled(-1.0,g);dg0=-g.norm2();cnt=0;}

        double alpha=1.0,En=E; bool ok=false;
        for (int ls=0;ls<MAX_LS_STEPS;++ls){
            xt.assign_sum(x,alpha,dd); En=eval(xt,gn);
            if (En<=E+ARMIJO_C1*alpha*dg0){ok=true;break;} alpha*=0.5; if(alpha<1e-15)break;}
        if (!ok){alpha=1e-8/std::max(1.0,g.norm());xt.assign_sum(x,-alpha,g);En=eval(xt,gn);cnt=0;}

        {Vec sn(dim),yn(dim);sn.assign_diff(xt,x);yn.assign_diff(gn,g);
         double sy=sn.dot(yn),ss=sn.norm2();
         if (sy>1e-10*ss){s[ptr].copy_from(sn);y[ptr].copy_from(yn);rho[ptr]=1.0/sy;
             ptr=(ptr+1)%m;cnt=std::min(cnt+1,m);}}
        x.copy_from(xt); g.copy_from(gn); E=En;
        if (E<=etol) return {E,true};
    }
    return {E, E<=etol};
}

// ===========================================================================
// Steric-clash PenaltyEvaluator (duplicated from _relax.cpp)
// ===========================================================================
struct RelaxCell {
    double inv_cell; int nx,ny,nz; double ox,oy,oz;
    std::vector<int> head, nxt;
    void build(const double* p, int n, double cs){
        inv_cell=1.0/cs;
        double xmn=p[0],xmx=p[0],ymn=p[1],ymx=p[1],zmn=p[2],zmx=p[2];
        for(int i=1;i<n;++i){
            xmn=std::min(xmn,p[i*3]);xmx=std::max(xmx,p[i*3]);
            ymn=std::min(ymn,p[i*3+1]);ymx=std::max(ymx,p[i*3+1]);
            zmn=std::min(zmn,p[i*3+2]);zmx=std::max(zmx,p[i*3+2]);}
        ox=xmn-cs;oy=ymn-cs;oz=zmn-cs;
        nx=int((xmx-ox)*inv_cell)+2;ny=int((ymx-oy)*inv_cell)+2;nz=int((zmx-oz)*inv_cell)+2;
        head.assign(std::size_t(nx*ny*nz),-1);nxt.resize(std::size_t(n));
        for(int i=0;i<n;++i){
            int cx=int((p[i*3]-ox)*inv_cell),cy=int((p[i*3+1]-oy)*inv_cell),cz=int((p[i*3+2]-oz)*inv_cell);
            int cid=cx+nx*(cy+ny*cz);nxt[i]=head[cid];head[cid]=i;}}
    template<typename F> void for_each_pair(F f) const {
        for(int cz=0;cz<nz;++cz)for(int cy=0;cy<ny;++cy)for(int cx=0;cx<nx;++cx){
            int cid=cx+nx*(cy+ny*cz);
            for(int i=head[cid];i>=0;i=nxt[i]){
                for(int j=nxt[i];j>=0;j=nxt[j])f(i,j);
                for(int dz=-1;dz<=1;++dz)for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
                    if(!dx&&!dy&&!dz)continue;
                    int ncx=cx+dx,ncy=cy+dy,ncz=cz+dz;
                    if(ncx<0||ncy<0||ncz<0||ncx>=nx||ncy>=ny||ncz>=nz)continue;
                    int nid=ncx+nx*(ncy+ny*ncz);if(nid<=cid)continue;
                    for(int k=head[nid];k>=0;k=nxt[k])f(i,k);}}}};
};

class PenaltyEvaluator {
    const double* rad_; double cs_; int n_; double cell_sz_;
    RelaxCell cl_;
public:
    PenaltyEvaluator(const double* r, double cs, int n): rad_(r),cs_(cs),n_(n){
        double mr=0; for(int i=0;i<n;++i)mr=std::max(mr,r[i]);
        cell_sz_=std::max(1e-6,cs_*2.0*mr);}
    double evaluate(const Vec& x, Vec& g){
        double E=0; g.zero();
        const double* xd=x.data(); double* gd=g.data();
        auto acc=[&](int i,int j){
            double dx=xd[3*i]-xd[3*j],dy=xd[3*i+1]-xd[3*j+1],dz=xd[3*i+2]-xd[3*j+2];
            double d2=dx*dx+dy*dy+dz*dz,thr=cs_*(rad_[i]+rad_[j]);
            if(d2>=thr*thr)return;
            double d=std::sqrt(d2),ov=thr-d; E+=0.5*ov*ov;
            if(d>1e-10){double gf=-ov/d,gx=gf*dx,gy=gf*dy,gz=gf*dz;
                gd[3*i]+=gx;gd[3*i+1]+=gy;gd[3*i+2]+=gz;
                gd[3*j]-=gx;gd[3*j+1]-=gy;gd[3*j+2]-=gz;}};
        if(n_<RELAX_CELL_THR){for(int i=0;i<n_-1;++i)for(int j=i+1;j<n_;++j)acc(i,j);}
        else{cl_.build(xd,n_,cell_sz_);cl_.for_each_pair(acc);}
        return E;}
};

// ===========================================================================
// Angular repulsion neighbour lists
// ===========================================================================
using CellKey = std::array<int,3>;
struct CKH {std::size_t operator()(const CellKey& k)const noexcept{
    std::size_t h=std::size_t(k[0]);
    h^=std::size_t(k[1])*2654435761ULL+0x9e3779b9ULL+(h<<6)+(h>>2);
    h^=std::size_t(k[2])*2246822519ULL+0x9e3779b9ULL+(h<<6)+(h>>2);return h;}};
using CellMap = std::unordered_map<CellKey,std::vector<int>,CKH>;

static std::vector<std::vector<int>> build_nb(const double* p, int n, double cut, bool use_cell){
    const double c2=cut*cut;
    const std::size_t nn=std::size_t(n);
    std::vector<std::vector<int>> nb(nn);
    if(!use_cell){
        for(int i=0;i<n;++i)for(int j=0;j<n;++j){if(j==i)continue;
            double dx=p[i*3]-p[j*3],dy=p[i*3+1]-p[j*3+1],dz=p[i*3+2]-p[j*3+2];
            if(dx*dx+dy*dy+dz*dz<=c2)nb[i].push_back(j);} return nb;}
    double ic=1.0/cut; CellMap cells; cells.reserve(std::size_t(n));
    for(int i=0;i<n;++i){CellKey k={int(std::floor(p[i*3]*ic)),int(std::floor(p[i*3+1]*ic)),int(std::floor(p[i*3+2]*ic))};cells[k].push_back(i);}
    for(int i=0;i<n;++i){
        CellKey ck={int(std::floor(p[i*3]*ic)),int(std::floor(p[i*3+1]*ic)),int(std::floor(p[i*3+2]*ic))};
        for(int dz=-1;dz<=1;++dz)for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
            CellKey nk={ck[0]+dx,ck[1]+dy,ck[2]+dz};auto it=cells.find(nk);if(it==cells.end())continue;
            for(int j:it->second){if(j==i)continue;
                double ddx=p[i*3]-p[j*3],ddy=p[i*3+1]-p[j*3+1],ddz=p[i*3+2]-p[j*3+2];
                if(ddx*ddx+ddy*ddy+ddz*ddz<=c2)nb[std::size_t(i)].push_back(j);}}}
    return nb;}

// ===========================================================================
// Angular energy + gradient
// U = Σ_i Σ_{j<k∈N(i)} 2/(1−cosθ_{jk}+ε)
// Gradient matches v0.1.14 accumulate_gradient (both j,k orderings summed).
// ===========================================================================
static double eval_angular(const double* p, int n,
    const std::vector<std::vector<int>>& nb, double* grad)
{
    double U = 0.0;
    if (grad) std::fill(grad, grad + 3*n, 0.0);

#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif
    // Thread-local gradient buffers to avoid false sharing
    std::vector<std::vector<double>> tgrad(
        static_cast<std::size_t>(nthreads),
        std::vector<double>(static_cast<std::size_t>(n * 3), 0.0));

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,16) reduction(+:U)
#endif
    for (int i = 0; i < n; ++i) {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        const auto& nbi = nb[static_cast<std::size_t>(i)];
        std::size_t nc = nbi.size();
        if (nc < 2) continue;

        std::vector<double> ux(nc), uy(nc), uz(nc), id(nc);
        for (std::size_t idx = 0; idx < nc; ++idx) {
            int j = nbi[idx];
            double dx = p[i*3]-p[j*3], dy = p[i*3+1]-p[j*3+1], dz = p[i*3+2]-p[j*3+2];
            double d = std::sqrt(dx*dx+dy*dy+dz*dz);
            if (d > 0) { double inv = 1.0/d; ux[idx]=dx*inv; uy[idx]=dy*inv; uz[idx]=dz*inv; id[idx]=inv; }
            else { ux[idx]=uy[idx]=uz[idx]=id[idx]=0; }
        }

        auto& tg = tgrad[static_cast<std::size_t>(tid)];
        for (std::size_t ji = 0; ji < nc; ++ji) {
            if (id[ji] <= 0) continue;
            double idj = id[ji], ujx = ux[ji], ujy = uy[ji], ujz = uz[ji];
            for (std::size_t ki = 0; ki < nc; ++ki) {
                double cv  = ujx*ux[ki]+ujy*uy[ki]+ujz*uz[ki];
                double den = 1.0 - cv + EPS;
                if (ki > ji) U += 2.0/den;
                if (grad) {
                    double w = 1.0/(den*den);
                    tg[i*3  ] += w*(ux[ki]-cv*ujx)*idj;
                    tg[i*3+1] += w*(uy[ki]-cv*ujy)*idj;
                    tg[i*3+2] += w*(uz[ki]-cv*ujz)*idj;
                }
            }
        }
    }

    // Merge thread-local gradients
    if (grad) {
        for (int t = 0; t < nthreads; ++t)
            for (int k = 0; k < n*3; ++k)
                grad[k] += tgrad[static_cast<std::size_t>(t)][k];
    }
    return U;
}

// ===========================================================================
// place_maxent_cpp
// ===========================================================================
F64Array place_maxent_cpp_impl(
    F64Array pts_in, F64Array radii_in,
    double cov_scale, double region_radius, double ang_cutoff,
    int maxent_steps, double trust_radius, double convergence_tol, long long seed)
{
    auto pb=pts_in.request(); auto rb=radii_in.request();
    const int n=int(pb.shape[0]);
    F64Array out({py::ssize_t(n),py::ssize_t(3)});
    double* outp=static_cast<double*>(out.request().ptr);
    const double* srcp=static_cast<const double*>(pb.ptr);
    std::copy(srcp,srcp+n*3,outp);
    const double* radii=static_cast<const double*>(rb.ptr);
    if(n<2)return out;

    std::mt19937_64 rng;
    if(seed<0){std::random_device rd;rng.seed(rd());}
    else{rng.seed(std::uint64_t(seed));}
    std::normal_distribution<double> ndist(0,1);

    PenaltyEvaluator relax_eval(radii,cov_scale,n);

    auto do_relax=[&](Vec& x,int mi){
        double mr=*std::max_element(radii,radii+n),js=1e-6*std::max(mr,1e-3);
        for(int i=0;i<n-1;++i)for(int j=i+1;j<n;++j){
            double dx=x[3*i]-x[3*j],dy=x[3*i+1]-x[3*j+1],dz=x[3*i+2]-x[3*j+2];
            if(dx*dx+dy*dy+dz*dz<1e-20)for(int d=0;d<3;++d){
                double jt=js*ndist(rng);x[3*i+d]+=jt;x[3*j+d]-=jt;}}
        lbfgs_minimize(x,[&](const Vec& p,Vec& g){return relax_eval.evaluate(p,g);},mi,1e-12);};

    const int dim=3*n,m=LBFGS_M;
    std::vector<Vec> sb(m,Vec(dim)),yb(m,Vec(dim));
    std::vector<double> rho(m,0),alp(m,0);
    int bptr=0,bcnt=0;
    Vec g(dim),gn(dim),q(dim),rl(dim),dd(dim),xt(dim);

    Vec x(dim); x.copy_from(outp,dim);
    // Initial CoM
    {double mx=0,my=0,mz=0;
     for(int i=0;i<n;++i){mx+=x[3*i];my+=x[3*i+1];mz+=x[3*i+2];}
     mx/=n;my/=n;mz/=n;
     for(int i=0;i<n;++i){x[3*i]-=mx;x[3*i+1]-=my;x[3*i+2]-=mz;}}

    const double k_rest=0.1*(trust_radius/0.5);

    for(int step=0;step<maxent_steps;++step){
        auto nb=build_nb(x.data(),n,ang_cutoff,n>=CELL_LIST_THRESHOLD);
        eval_angular(x.data(),n,nb,g.data());

        // L-BFGS direction
        q.copy_from(g);
        for(int i=0;i<bcnt;++i){int k=(bptr+m-1-i)%m;alp[i]=rho[k]*sb[k].dot(q);q.add_scaled(-alp[i],yb[k]);}
        if(bcnt>0){int k=(bptr+m-1)%m;double sy=sb[k].dot(yb[k]),yy=yb[k].norm2();
            rl.assign_scaled(yy>1e-20?sy/yy:1.0,q);}
        else{double gn2=g.norm();if(gn2>1e-20)rl.assign_scaled(1.0/gn2,q);else rl.copy_from(q);}
        for(int i=bcnt-1;i>=0;--i){int k=(bptr+m-1-i)%m;double b=rho[k]*yb[k].dot(rl);rl.add_scaled(alp[i]-b,sb[k]);}
        dd.assign_scaled(-1.0,rl);
        if(dd.dot(g)>=-1e-14*dd.norm()*g.norm()){dd.assign_scaled(-1.0,g);bcnt=0;}

        // Per-atom trust-radius clamp (uniform rescaling)
        {double mdisp=0;
         for(int i=0;i<n;++i){double d2=dd[3*i]*dd[3*i]+dd[3*i+1]*dd[3*i+1]+dd[3*i+2]*dd[3*i+2];mdisp=std::max(mdisp,std::sqrt(d2));}
         if(mdisp>trust_radius){double sc=trust_radius/mdisp;dd.assign_scaled(sc,dd);}}

        // Step (dd is descent direction: x += dd minimises angular repulsion)
        x.add_scaled(1.0,dd);

        // Soft restoring force
        for(int i=0;i<n;++i){
            double rx=x[3*i],ry=x[3*i+1],rz=x[3*i+2];
            double r=std::sqrt(rx*rx+ry*ry+rz*rz),ex=std::max(0.0,r-region_radius);
            if(ex>0){double sr=std::max(r,1e-10);
                x[3*i  ]-=k_rest*ex*rx/sr;x[3*i+1]-=k_rest*ex*ry/sr;x[3*i+2]-=k_rest*ex*rz/sr;}}

        // CoM pinning
        {double mx=0,my=0,mz=0;
         for(int i=0;i<n;++i){mx+=x[3*i];my+=x[3*i+1];mz+=x[3*i+2];}
         mx/=n;my/=n;mz/=n;
         for(int i=0;i<n;++i){x[3*i]-=mx;x[3*i+1]-=my;x[3*i+2]-=mz;}}

        // Relax
        do_relax(x,50);

        // History update: evaluate gradient after step, update s/y buffers
        {auto nb2=build_nb(x.data(),n,ang_cutoff,n>=CELL_LIST_THRESHOLD);
         eval_angular(x.data(),n,nb2,gn.data());
         Vec sn(dim),yn(dim);sn.copy_from(dd);yn.assign_diff(gn,g);
         double sy=sn.dot(yn),ss=sn.norm2();
         if(sy>1e-10*ss){sb[bptr].copy_from(sn);yb[bptr].copy_from(yn);rho[bptr]=1.0/sy;
             bptr=(bptr+1)%m;bcnt=std::min(bcnt+1,m);}}

        // Early termination: converged when RMS gradient per atom < tol
        if(convergence_tol > 0.0) {
            double gnorm2 = 0.0;
            for(int i=0;i<dim;++i) gnorm2 += gn[i]*gn[i];
            double rms = std::sqrt(gnorm2 / n);
            if(rms < convergence_tol) break;
        }
    }

    x.copy_to(outp); return out;
}

// ===========================================================================
// angular_repulsion_gradient (v0.1.14 compatible, unchanged API)
// ===========================================================================
F64Array angular_repulsion_gradient_cpp(F64Array pts_in, double cutoff){
    auto buf=pts_in.request();
    const int n=int(buf.shape[0]); const double* p=static_cast<const double*>(buf.ptr);
    F64Array grad_out({py::ssize_t(n),py::ssize_t(3)});
    double* g=static_cast<double*>(grad_out.request().ptr);
    std::fill(g,g+n*3,0.0);
    auto nb=build_nb(p,n,cutoff,n>=CELL_LIST_THRESHOLD);
    eval_angular(p,n,nb,g);
    return grad_out;}

// ===========================================================================
PYBIND11_MODULE(_maxent_core, m) {
    m.doc()=
        "pasted._ext._maxent_core (v0.1.15)\n"
        "angular_repulsion_gradient(pts, cutoff) -> grad ndarray(n,3)\n"
        "place_maxent_cpp(pts, radii, cov_scale, region_radius, ang_cutoff,\n"
        "                 maxent_steps, trust_radius=0.5, convergence_tol=1e-3, seed=-1) -> ndarray(n,3)\n";

    m.def("angular_repulsion_gradient",&angular_repulsion_gradient_cpp,
        py::arg("pts"),py::arg("cutoff"),
        "Gradient of angular repulsion potential. Cell List for N>=32.");

    m.def("place_maxent_cpp",&place_maxent_cpp_impl,
        py::arg("pts"),py::arg("radii"),
        py::arg("cov_scale"),py::arg("region_radius"),py::arg("ang_cutoff"),
        py::arg("maxent_steps"),py::arg("trust_radius")=0.5,
        py::arg("convergence_tol")=1e-3,py::arg("seed")=-1LL,
        R"(Full maxent gradient-descent loop in C++ with L-BFGS and trust-radius cap.

Parameters
----------
pts           : (n,3) float64 – initial positions (Ang)
radii         : (n,)  float64 – covalent radii (Ang)
cov_scale     : float          – minimum distance scale factor
region_radius : float          – soft-restoring-force sphere radius (Ang)
ang_cutoff    : float          – angular-repulsion neighbour cutoff (Ang)
maxent_steps  : int            – number of L-BFGS outer iterations
trust_radius  : float          – per-atom max displacement per step (Ang, default 0.5)
seed          : int            – RNG seed for coincident-atom jitter; -1 = random

Returns
-------
pts_out : (n,3) float64)");
}
