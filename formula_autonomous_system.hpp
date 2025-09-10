/**
 * @file formula_autonomous_system.hpp
 * @author Jiwon Seok (jiwonseok@hanyang.ac.kr)
 * @author MinKyu Cho (chomk2000@hanyang.ac.kr)
 * @brief 
 * @version 0.1
 * @date 2025-07-21
 * 
 * @copyright Copyright (c) 2025
 */

#ifndef FORMULA_AUTONOMOUS_SYSTEM_HPP
#define FORMULA_AUTONOMOUS_SYSTEM_HPP

// C++
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <random>
#include <cmath>
#include <deque>

// spline.h
#ifndef TK_SPLINE_H
#define TK_SPLINE_H

#include <cstdio>
#include <cassert>
#include <cmath>
#include <vector>
#include <algorithm>
#ifdef HAVE_SSTREAM
#include <sstream>
#include <string>
#endif // HAVE_SSTREAM

// not ideal but disable unused-function warnings
// (we get them because we have implementations in the header file,
// and this is because we want to be able to quickly separate them
// into a cpp file if necessary)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

// unnamed namespace only because the implementation is in this
// header file and we don't want to export symbols to the obj files
namespace
{

namespace tk
{

// spline interpolation
class spline
{
public:
    // spline types
    enum spline_type {
        linear = 10,            // linear interpolation
        cspline = 30,           // cubic splines (classical C^2)
        cspline_hermite = 31    // cubic hermite splines (local, only C^1)
    };

    // boundary condition type for the spline end-points
    enum bd_type {
        first_deriv = 1,
        second_deriv = 2,
        not_a_knot = 3
    };

protected:
    std::vector<double> m_x,m_y;            // x,y coordinates of points
    // interpolation parameters
    // f(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3
    // where a_i = y_i, or else it won't go through grid points
    std::vector<double> m_b,m_c,m_d;        // spline coefficients
    double m_c0;                            // for left extrapolation
    spline_type m_type;
    bd_type m_left, m_right;
    double  m_left_value, m_right_value;
    bool m_made_monotonic;
    void set_coeffs_from_b();               // calculate c_i, d_i from b_i
    size_t find_closest(double x) const;    // closest idx so that m_x[idx]<=x

public:
    // default constructor: set boundary condition to be zero curvature
    // at both ends, i.e. natural splines
    spline(): m_type(cspline),
        m_left(second_deriv), m_right(second_deriv),
        m_left_value(0.0), m_right_value(0.0), m_made_monotonic(false)
    {
        ;
    }
    spline(const std::vector<double>& X, const std::vector<double>& Y,
           spline_type type = cspline,
           bool make_monotonic = false,
           bd_type left  = second_deriv, double left_value  = 0.0,
           bd_type right = second_deriv, double right_value = 0.0
          ):
        m_type(type),
        m_left(left), m_right(right),
        m_left_value(left_value), m_right_value(right_value),
        m_made_monotonic(false) // false correct here: make_monotonic() sets it
    {
        this->set_points(X,Y,m_type);
        if(make_monotonic) {
            this->make_monotonic();
        }
    }


    // modify boundary conditions: if called it must be before set_points()
    void set_boundary(bd_type left, double left_value,
                      bd_type right, double right_value);

    // set all data points (cubic_spline=false means linear interpolation)
    void set_points(const std::vector<double>& x,
                    const std::vector<double>& y,
                    spline_type type=cspline);

    // adjust coefficients so that the spline becomes piecewise monotonic
    // where possible
    //   this is done by adjusting slopes at grid points by a non-negative
    //   factor and this will break C^2
    //   this can also break boundary conditions if adjustments need to
    //   be made at the boundary points
    // returns false if no adjustments have been made, true otherwise
    bool make_monotonic();

    // evaluates the spline at point x
    double operator() (double x) const;
    double deriv(int order, double x) const;

    // solves for all x so that: spline(x) = y
    std::vector<double> solve(double y, bool ignore_extrapolation=true) const;

    // returns the input data points
    std::vector<double> get_x() const { return m_x; }
    std::vector<double> get_y() const { return m_y; }
    double get_x_min() const { assert(!m_x.empty()); return m_x.front(); }
    double get_x_max() const { assert(!m_x.empty()); return m_x.back(); }

#ifdef HAVE_SSTREAM
    // spline info string, i.e. spline type, boundary conditions etc.
    std::string info() const;
#endif // HAVE_SSTREAM

};



namespace internal
{

// band matrix solver
class band_matrix
{
private:
    std::vector< std::vector<double> > m_upper;  // upper band
    std::vector< std::vector<double> > m_lower;  // lower band
public:
    band_matrix() {};                             // constructor
    band_matrix(int dim, int n_u, int n_l);       // constructor
    ~band_matrix() {};                            // destructor
    void resize(int dim, int n_u, int n_l);      // init with dim,n_u,n_l
    int dim() const;                             // matrix dimension
    int num_upper() const
    {
        return (int)m_upper.size()-1;
    }
    int num_lower() const
    {
        return (int)m_lower.size()-1;
    }
    // access operator
    double & operator () (int i, int j);            // write
    double   operator () (int i, int j) const;      // read
    // we can store an additional diagonal (in m_lower)
    double& saved_diag(int i);
    double  saved_diag(int i) const;
    void lu_decompose();
    std::vector<double> r_solve(const std::vector<double>& b) const;
    std::vector<double> l_solve(const std::vector<double>& b) const;
    std::vector<double> lu_solve(const std::vector<double>& b,
                                 bool is_lu_decomposed=false);

};

double get_eps();

std::vector<double> solve_cubic(double a, double b, double c, double d,
                                int newton_iter=0);

} // namespace internal




// ---------------------------------------------------------------------
// implementation part, which could be separated into a cpp file
// ---------------------------------------------------------------------

// spline implementation
// -----------------------

void spline::set_boundary(spline::bd_type left, double left_value,
                          spline::bd_type right, double right_value)
{
    assert(m_x.size()==0);          // set_points() must not have happened yet
    m_left=left;
    m_right=right;
    m_left_value=left_value;
    m_right_value=right_value;
}


void spline::set_coeffs_from_b()
{
    assert(m_x.size()==m_y.size());
    assert(m_x.size()==m_b.size());
    assert(m_x.size()>2);
    size_t n=m_b.size();
    if(m_c.size()!=n)
        m_c.resize(n);
    if(m_d.size()!=n)
        m_d.resize(n);

    for(size_t i=0; i<n-1; i++) {
        const double h  = m_x[i+1]-m_x[i];
        // from continuity and differentiability condition
        m_c[i] = ( 3.0*(m_y[i+1]-m_y[i])/h - (2.0*m_b[i]+m_b[i+1]) ) / h;
        // from differentiability condition
        m_d[i] = ( (m_b[i+1]-m_b[i])/(3.0*h) - 2.0/3.0*m_c[i] ) / h;
    }

    // for left extrapolation coefficients
    m_c0 = (m_left==first_deriv) ? 0.0 : m_c[0];
}

void spline::set_points(const std::vector<double>& x,
                        const std::vector<double>& y,
                        spline_type type)
{
    assert(x.size()==y.size());
    assert(x.size()>=3);
    // not-a-knot with 3 points has many solutions
    if(m_left==not_a_knot || m_right==not_a_knot)
        assert(x.size()>=4);
    m_type=type;
    m_made_monotonic=false;
    m_x=x;
    m_y=y;
    int n = (int) x.size();
    // check strict monotonicity of input vector x
    for(int i=0; i<n-1; i++) {
        assert(m_x[i]<m_x[i+1]);
    }


    if(type==linear) {
        // linear interpolation
        m_d.resize(n);
        m_c.resize(n);
        m_b.resize(n);
        for(int i=0; i<n-1; i++) {
            m_d[i]=0.0;
            m_c[i]=0.0;
            m_b[i]=(m_y[i+1]-m_y[i])/(m_x[i+1]-m_x[i]);
        }
        // ignore boundary conditions, set slope equal to the last segment
        m_b[n-1]=m_b[n-2];
        m_c[n-1]=0.0;
        m_d[n-1]=0.0;
    } else if(type==cspline) {
        // classical cubic splines which are C^2 (twice cont differentiable)
        // this requires solving an equation system

        // setting up the matrix and right hand side of the equation system
        // for the parameters b[]
        int n_upper = (m_left  == spline::not_a_knot) ? 2 : 1;
        int n_lower = (m_right == spline::not_a_knot) ? 2 : 1;
        internal::band_matrix A(n,n_upper,n_lower);
        std::vector<double>  rhs(n);
        for(int i=1; i<n-1; i++) {
            A(i,i-1)=1.0/3.0*(x[i]-x[i-1]);
            A(i,i)=2.0/3.0*(x[i+1]-x[i-1]);
            A(i,i+1)=1.0/3.0*(x[i+1]-x[i]);
            rhs[i]=(y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1]);
        }
        // boundary conditions
        if(m_left == spline::second_deriv) {
            // 2*c[0] = f''
            A(0,0)=2.0;
            A(0,1)=0.0;
            rhs[0]=m_left_value;
        } else if(m_left == spline::first_deriv) {
            // b[0] = f', needs to be re-expressed in terms of c:
            // (2c[0]+c[1])(x[1]-x[0]) = 3 ((y[1]-y[0])/(x[1]-x[0]) - f')
            A(0,0)=2.0*(x[1]-x[0]);
            A(0,1)=1.0*(x[1]-x[0]);
            rhs[0]=3.0*((y[1]-y[0])/(x[1]-x[0])-m_left_value);
        } else if(m_left == spline::not_a_knot) {
            // f'''(x[1]) exists, i.e. d[0]=d[1], or re-expressed in c:
            // -h1*c[0] + (h0+h1)*c[1] - h0*c[2] = 0
            A(0,0) = -(x[2]-x[1]);
            A(0,1) = x[2]-x[0];
            A(0,2) = -(x[1]-x[0]);
            rhs[0] = 0.0;
        } else {
            assert(false);
        }
        if(m_right == spline::second_deriv) {
            // 2*c[n-1] = f''
            A(n-1,n-1)=2.0;
            A(n-1,n-2)=0.0;
            rhs[n-1]=m_right_value;
        } else if(m_right == spline::first_deriv) {
            // b[n-1] = f', needs to be re-expressed in terms of c:
            // (c[n-2]+2c[n-1])(x[n-1]-x[n-2])
            // = 3 (f' - (y[n-1]-y[n-2])/(x[n-1]-x[n-2]))
            A(n-1,n-1)=2.0*(x[n-1]-x[n-2]);
            A(n-1,n-2)=1.0*(x[n-1]-x[n-2]);
            rhs[n-1]=3.0*(m_right_value-(y[n-1]-y[n-2])/(x[n-1]-x[n-2]));
        } else if(m_right == spline::not_a_knot) {
            // f'''(x[n-2]) exists, i.e. d[n-3]=d[n-2], or re-expressed in c:
            // -h_{n-2}*c[n-3] + (h_{n-3}+h_{n-2})*c[n-2] - h_{n-3}*c[n-1] = 0
            A(n-1,n-3) = -(x[n-1]-x[n-2]);
            A(n-1,n-2) = x[n-1]-x[n-3];
            A(n-1,n-1) = -(x[n-2]-x[n-3]);
            rhs[0] = 0.0;
        } else {
            assert(false);
        }

        // solve the equation system to obtain the parameters c[]
        m_c=A.lu_solve(rhs);

        // calculate parameters b[] and d[] based on c[]
        m_d.resize(n);
        m_b.resize(n);
        for(int i=0; i<n-1; i++) {
            m_d[i]=1.0/3.0*(m_c[i+1]-m_c[i])/(x[i+1]-x[i]);
            m_b[i]=(y[i+1]-y[i])/(x[i+1]-x[i])
                   - 1.0/3.0*(2.0*m_c[i]+m_c[i+1])*(x[i+1]-x[i]);
        }
        // for the right extrapolation coefficients (zero cubic term)
        // f_{n-1}(x) = y_{n-1} + b*(x-x_{n-1}) + c*(x-x_{n-1})^2
        double h=x[n-1]-x[n-2];
        // m_c[n-1] is determined by the boundary condition
        m_d[n-1]=0.0;
        m_b[n-1]=3.0*m_d[n-2]*h*h+2.0*m_c[n-2]*h+m_b[n-2];   // = f'_{n-2}(x_{n-1})
        if(m_right==first_deriv)
            m_c[n-1]=0.0;   // force linear extrapolation

    } else if(type==cspline_hermite) {
        // hermite cubic splines which are C^1 (cont. differentiable)
        // and derivatives are specified on each grid point
        // (here we use 3-point finite differences)
        m_b.resize(n);
        m_c.resize(n);
        m_d.resize(n);
        // set b to match 1st order derivative finite difference
        for(int i=1; i<n-1; i++) {
            const double h  = m_x[i+1]-m_x[i];
            const double hl = m_x[i]-m_x[i-1];
            m_b[i] = -h/(hl*(hl+h))*m_y[i-1] + (h-hl)/(hl*h)*m_y[i]
                     +  hl/(h*(hl+h))*m_y[i+1];
        }
        // boundary conditions determine b[0] and b[n-1]
        if(m_left==first_deriv) {
            m_b[0]=m_left_value;
        } else if(m_left==second_deriv) {
            const double h = m_x[1]-m_x[0];
            m_b[0]=0.5*(-m_b[1]-0.5*m_left_value*h+3.0*(m_y[1]-m_y[0])/h);
        } else if(m_left == not_a_knot) {
            // f''' continuous at x[1]
            const double h0 = m_x[1]-m_x[0];
            const double h1 = m_x[2]-m_x[1];
            m_b[0]= -m_b[1] + 2.0*(m_y[1]-m_y[0])/h0
                    + h0*h0/(h1*h1)*(m_b[1]+m_b[2]-2.0*(m_y[2]-m_y[1])/h1);
        } else {
            assert(false);
        }
        if(m_right==first_deriv) {
            m_b[n-1]=m_right_value;
            m_c[n-1]=0.0;
        } else if(m_right==second_deriv) {
            const double h = m_x[n-1]-m_x[n-2];
            m_b[n-1]=0.5*(-m_b[n-2]+0.5*m_right_value*h+3.0*(m_y[n-1]-m_y[n-2])/h);
            m_c[n-1]=0.5*m_right_value;
        } else if(m_right == not_a_knot) {
            // f''' continuous at x[n-2]
            const double h0 = m_x[n-2]-m_x[n-3];
            const double h1 = m_x[n-1]-m_x[n-2];
            m_b[n-1]= -m_b[n-2] + 2.0*(m_y[n-1]-m_y[n-2])/h1 + h1*h1/(h0*h0)
                      *(m_b[n-3]+m_b[n-2]-2.0*(m_y[n-2]-m_y[n-3])/h0);
            // f'' continuous at x[n-1]: c[n-1] = 3*d[n-2]*h[n-2] + c[n-1]
            m_c[n-1]=(m_b[n-2]+2.0*m_b[n-1])/h1-3.0*(m_y[n-1]-m_y[n-2])/(h1*h1);
        } else {
            assert(false);
        }
        m_d[n-1]=0.0;

        // parameters c and d are determined by continuity and differentiability
        set_coeffs_from_b();

    } else {
        assert(false);
    }

    // for left extrapolation coefficients
    m_c0 = (m_left==first_deriv) ? 0.0 : m_c[0];
}

bool spline::make_monotonic()
{
    assert(m_x.size()==m_y.size());
    assert(m_x.size()==m_b.size());
    assert(m_x.size()>2);
    bool modified = false;
    const int n=(int)m_x.size();
    // make sure: input data monotonic increasing --> b_i>=0
    //            input data monotonic decreasing --> b_i<=0
    for(int i=0; i<n; i++) {
        int im1 = std::max(i-1, 0);
        int ip1 = std::min(i+1, n-1);
        if( ((m_y[im1]<=m_y[i]) && (m_y[i]<=m_y[ip1]) && m_b[i]<0.0) ||
            ((m_y[im1]>=m_y[i]) && (m_y[i]>=m_y[ip1]) && m_b[i]>0.0) ) {
            modified=true;
            m_b[i]=0.0;
        }
    }
    // if input data is monotonic (b[i], b[i+1], avg have all the same sign)
    // ensure a sufficient criteria for monotonicity is satisfied:
    //     sqrt(b[i]^2+b[i+1]^2) <= 3 |avg|, with avg=(y[i+1]-y[i])/h,
    for(int i=0; i<n-1; i++) {
        double h = m_x[i+1]-m_x[i];
        double avg = (m_y[i+1]-m_y[i])/h;
        if( avg==0.0 && (m_b[i]!=0.0 || m_b[i+1]!=0.0) ) {
            modified=true;
            m_b[i]=0.0;
            m_b[i+1]=0.0;
        } else if( (m_b[i]>=0.0 && m_b[i+1]>=0.0 && avg>0.0) ||
                   (m_b[i]<=0.0 && m_b[i+1]<=0.0 && avg<0.0) ) {
            // input data is monotonic
            double r = sqrt(m_b[i]*m_b[i]+m_b[i+1]*m_b[i+1])/std::fabs(avg);
            if(r>3.0) {
                // sufficient criteria for monotonicity: r<=3
                // adjust b[i] and b[i+1]
                modified=true;
                m_b[i]   *= (3.0/r);
                m_b[i+1] *= (3.0/r);
            }
        }
    }

    if(modified==true) {
        set_coeffs_from_b();
        m_made_monotonic=true;
    }

    return modified;
}

// return the closest idx so that m_x[idx] <= x (return 0 if x<m_x[0])
size_t spline::find_closest(double x) const
{
    std::vector<double>::const_iterator it;
    it=std::upper_bound(m_x.begin(),m_x.end(),x);       // *it > x
    size_t idx = std::max( int(it-m_x.begin())-1, 0);   // m_x[idx] <= x
    return idx;
}

double spline::operator() (double x) const
{
    // polynomial evaluation using Horner's scheme
    // TODO: consider more numerically accurate algorithms, e.g.:
    //   - Clenshaw
    //   - Even-Odd method by A.C.R. Newbery
    //   - Compensated Horner Scheme
    size_t n=m_x.size();
    size_t idx=find_closest(x);

    double h=x-m_x[idx];
    double interpol;
    if(x<m_x[0]) {
        // extrapolation to the left
        interpol=(m_c0*h + m_b[0])*h + m_y[0];
    } else if(x>m_x[n-1]) {
        // extrapolation to the right
        interpol=(m_c[n-1]*h + m_b[n-1])*h + m_y[n-1];
    } else {
        // interpolation
        interpol=((m_d[idx]*h + m_c[idx])*h + m_b[idx])*h + m_y[idx];
    }
    return interpol;
}

double spline::deriv(int order, double x) const
{
    assert(order>0);
    size_t n=m_x.size();
    size_t idx = find_closest(x);

    double h=x-m_x[idx];
    double interpol;
    if(x<m_x[0]) {
        // extrapolation to the left
        switch(order) {
        case 1:
            interpol=2.0*m_c0*h + m_b[0];
            break;
        case 2:
            interpol=2.0*m_c0;
            break;
        default:
            interpol=0.0;
            break;
        }
    } else if(x>m_x[n-1]) {
        // extrapolation to the right
        switch(order) {
        case 1:
            interpol=2.0*m_c[n-1]*h + m_b[n-1];
            break;
        case 2:
            interpol=2.0*m_c[n-1];
            break;
        default:
            interpol=0.0;
            break;
        }
    } else {
        // interpolation
        switch(order) {
        case 1:
            interpol=(3.0*m_d[idx]*h + 2.0*m_c[idx])*h + m_b[idx];
            break;
        case 2:
            interpol=6.0*m_d[idx]*h + 2.0*m_c[idx];
            break;
        case 3:
            interpol=6.0*m_d[idx];
            break;
        default:
            interpol=0.0;
            break;
        }
    }
    return interpol;
}

std::vector<double> spline::solve(double y, bool ignore_extrapolation) const
{
    std::vector<double> x;          // roots for the entire spline
    std::vector<double> root;       // roots for each piecewise cubic
    const size_t n=m_x.size();

    // left extrapolation
    if(ignore_extrapolation==false) {
        root = internal::solve_cubic(m_y[0]-y,m_b[0],m_c0,0.0,1);
        for(size_t j=0; j<root.size(); j++) {
            if(root[j]<0.0) {
                x.push_back(m_x[0]+root[j]);
            }
        }
    }

    // brute force check if piecewise cubic has roots in their resp. segment
    // TODO: make more efficient
    for(size_t i=0; i<n-1; i++) {
        root = internal::solve_cubic(m_y[i]-y,m_b[i],m_c[i],m_d[i],1);
        for(size_t j=0; j<root.size(); j++) {
            double h = (i>0) ? (m_x[i]-m_x[i-1]) : 0.0;
            double eps = internal::get_eps()*512.0*std::min(h,1.0);
            if( (-eps<=root[j]) && (root[j]<m_x[i+1]-m_x[i]) ) {
                double new_root = m_x[i]+root[j];
                if(x.size()>0 && x.back()+eps > new_root) {
                    x.back()=new_root;      // avoid spurious duplicate roots
                } else {
                    x.push_back(new_root);
                }
            }
        }
    }

    // right extrapolation
    if(ignore_extrapolation==false) {
        root = internal::solve_cubic(m_y[n-1]-y,m_b[n-1],m_c[n-1],0.0,1);
        for(size_t j=0; j<root.size(); j++) {
            if(0.0<=root[j]) {
                x.push_back(m_x[n-1]+root[j]);
            }
        }
    }

    return x;
};


#ifdef HAVE_SSTREAM
std::string spline::info() const
{
    std::stringstream ss;
    ss << "type " << m_type << ", left boundary deriv " << m_left << " = ";
    ss << m_left_value << ", right boundary deriv " << m_right << " = ";
    ss << m_right_value << std::endl;
    if(m_made_monotonic) {
        ss << "(spline has been adjusted for piece-wise monotonicity)";
    }
    return ss.str();
}
#endif // HAVE_SSTREAM


namespace internal
{

// band_matrix implementation
// -------------------------

band_matrix::band_matrix(int dim, int n_u, int n_l)
{
    resize(dim, n_u, n_l);
}
void band_matrix::resize(int dim, int n_u, int n_l)
{
    assert(dim>0);
    assert(n_u>=0);
    assert(n_l>=0);
    m_upper.resize(n_u+1);
    m_lower.resize(n_l+1);
    for(size_t i=0; i<m_upper.size(); i++) {
        m_upper[i].resize(dim);
    }
    for(size_t i=0; i<m_lower.size(); i++) {
        m_lower[i].resize(dim);
    }
}
int band_matrix::dim() const
{
    if(m_upper.size()>0) {
        return m_upper[0].size();
    } else {
        return 0;
    }
}


// defines the new operator (), so that we can access the elements
// by A(i,j), index going from i=0,...,dim()-1
double & band_matrix::operator () (int i, int j)
{
    int k=j-i;       // what band is the entry
    assert( (i>=0) && (i<dim()) && (j>=0) && (j<dim()) );
    assert( (-num_lower()<=k) && (k<=num_upper()) );
    // k=0 -> diagonal, k<0 lower left part, k>0 upper right part
    if(k>=0)    return m_upper[k][i];
    else        return m_lower[-k][i];
}
double band_matrix::operator () (int i, int j) const
{
    int k=j-i;       // what band is the entry
    assert( (i>=0) && (i<dim()) && (j>=0) && (j<dim()) );
    assert( (-num_lower()<=k) && (k<=num_upper()) );
    // k=0 -> diagonal, k<0 lower left part, k>0 upper right part
    if(k>=0)    return m_upper[k][i];
    else        return m_lower[-k][i];
}
// second diag (used in LU decomposition), saved in m_lower
double band_matrix::saved_diag(int i) const
{
    assert( (i>=0) && (i<dim()) );
    return m_lower[0][i];
}
double & band_matrix::saved_diag(int i)
{
    assert( (i>=0) && (i<dim()) );
    return m_lower[0][i];
}

// LR-Decomposition of a band matrix
void band_matrix::lu_decompose()
{
    int  i_max,j_max;
    int  j_min;
    double x;

    // preconditioning
    // normalize column i so that a_ii=1
    for(int i=0; i<this->dim(); i++) {
        assert(this->operator()(i,i)!=0.0);
        this->saved_diag(i)=1.0/this->operator()(i,i);
        j_min=std::max(0,i-this->num_lower());
        j_max=std::min(this->dim()-1,i+this->num_upper());
        for(int j=j_min; j<=j_max; j++) {
            this->operator()(i,j) *= this->saved_diag(i);
        }
        this->operator()(i,i)=1.0;          // prevents rounding errors
    }

    // Gauss LR-Decomposition
    for(int k=0; k<this->dim(); k++) {
        i_max=std::min(this->dim()-1,k+this->num_lower());  // num_lower not a mistake!
        for(int i=k+1; i<=i_max; i++) {
            assert(this->operator()(k,k)!=0.0);
            x=-this->operator()(i,k)/this->operator()(k,k);
            this->operator()(i,k)=-x;                         // assembly part of L
            j_max=std::min(this->dim()-1,k+this->num_upper());
            for(int j=k+1; j<=j_max; j++) {
                // assembly part of R
                this->operator()(i,j)=this->operator()(i,j)+x*this->operator()(k,j);
            }
        }
    }
}
// solves Ly=b
std::vector<double> band_matrix::l_solve(const std::vector<double>& b) const
{
    assert( this->dim()==(int)b.size() );
    std::vector<double> x(this->dim());
    int j_start;
    double sum;
    for(int i=0; i<this->dim(); i++) {
        sum=0;
        j_start=std::max(0,i-this->num_lower());
        for(int j=j_start; j<i; j++) sum += this->operator()(i,j)*x[j];
        x[i]=(b[i]*this->saved_diag(i)) - sum;
    }
    return x;
}
// solves Rx=y
std::vector<double> band_matrix::r_solve(const std::vector<double>& b) const
{
    assert( this->dim()==(int)b.size() );
    std::vector<double> x(this->dim());
    int j_stop;
    double sum;
    for(int i=this->dim()-1; i>=0; i--) {
        sum=0;
        j_stop=std::min(this->dim()-1,i+this->num_upper());
        for(int j=i+1; j<=j_stop; j++) sum += this->operator()(i,j)*x[j];
        x[i]=( b[i] - sum ) / this->operator()(i,i);
    }
    return x;
}

std::vector<double> band_matrix::lu_solve(const std::vector<double>& b,
        bool is_lu_decomposed)
{
    assert( this->dim()==(int)b.size() );
    std::vector<double>  x,y;
    if(is_lu_decomposed==false) {
        this->lu_decompose();
    }
    y=this->l_solve(b);
    x=this->r_solve(y);
    return x;
}

// machine precision of a double, i.e. the successor of 1 is 1+eps
double get_eps()
{
    //return std::numeric_limits<double>::epsilon();    // __DBL_EPSILON__
    return 2.2204460492503131e-16;                      // 2^-52
}

// solutions for a + b*x = 0
std::vector<double> solve_linear(double a, double b)
{
    std::vector<double> x;      // roots
    if(b==0.0) {
        if(a==0.0) {
            // 0*x = 0
            x.resize(1);
            x[0] = 0.0;   // any x solves it but we need to pick one
            return x;
        } else {
            // 0*x + ... = 0, no solution
            return x;
        }
    } else {
        x.resize(1);
        x[0] = -a/b;
        return x;
    }
}

// solutions for a + b*x + c*x^2 = 0
std::vector<double> solve_quadratic(double a, double b, double c,
                                    int newton_iter=0)
{
    if(c==0.0) {
        return solve_linear(a,b);
    }
    // rescale so that we solve x^2 + 2p x + q = (x+p)^2 + q - p^2 = 0
    double p=0.5*b/c;
    double q=a/c;
    double discr = p*p-q;
    const double eps=0.5*internal::get_eps();
    double discr_err = (6.0*(p*p)+3.0*fabs(q)+fabs(discr))*eps;

    std::vector<double> x;      // roots
    if(fabs(discr)<=discr_err) {
        // discriminant is zero --> one root
        x.resize(1);
        x[0] = -p;
    } else if(discr<0) {
        // no root
    } else {
        // two roots
        x.resize(2);
        x[0] = -p - sqrt(discr);
        x[1] = -p + sqrt(discr);
    }

    // improve solution via newton steps
    for(size_t i=0; i<x.size(); i++) {
        for(int k=0; k<newton_iter; k++) {
            double f  = (c*x[i] + b)*x[i] + a;
            double f1 = 2.0*c*x[i] + b;
            // only adjust if slope is large enough
            if(fabs(f1)>1e-8) {
                x[i] -= f/f1;
            }
        }
    }

    return x;
}

// solutions for the cubic equation: a + b*x +c*x^2 + d*x^3 = 0
// this is a naive implementation of the analytic solution without
// optimisation for speed or numerical accuracy
// newton_iter: number of newton iterations to improve analytical solution
// see also
//   gsl: gsl_poly_solve_cubic() in solve_cubic.c
//   octave: roots.m - via eigenvalues of the Frobenius companion matrix
std::vector<double> solve_cubic(double a, double b, double c, double d,
                                int newton_iter)
{
    if(d==0.0) {
        return solve_quadratic(a,b,c,newton_iter);
    }

    // convert to normalised form: a + bx + cx^2 + x^3 = 0
    if(d!=1.0) {
        a/=d;
        b/=d;
        c/=d;
    }

    // convert to depressed cubic: z^3 - 3pz - 2q = 0
    // via substitution: z = x + c/3
    std::vector<double> z;              // roots of the depressed cubic
    double p = -(1.0/3.0)*b + (1.0/9.0)*(c*c);
    double r = 2.0*(c*c)-9.0*b;
    double q = -0.5*a - (1.0/54.0)*(c*r);
    double discr=p*p*p-q*q;             // discriminant
    // calculating numerical round-off errors with assumptions:
    //  - each operation is precise but each intermediate result x
    //    when stored has max error of x*eps
    //  - only multiplication with a power of 2 introduces no new error
    //  - a,b,c,d and some fractions (e.g. 1/3) have rounding errors eps
    //  - p_err << |p|, q_err << |q|, ... (this is violated in rare cases)
    // would be more elegant to use boost::numeric::interval<double>
    const double eps = internal::get_eps();
    double p_err = eps*((3.0/3.0)*fabs(b)+(4.0/9.0)*(c*c)+fabs(p));
    double r_err = eps*(6.0*(c*c)+18.0*fabs(b)+fabs(r));
    double q_err = 0.5*fabs(a)*eps + (1.0/54.0)*fabs(c)*(r_err+fabs(r)*3.0*eps)
                   + fabs(q)*eps;
    double discr_err = (p*p) * (3.0*p_err + fabs(p)*2.0*eps)
                       + fabs(q) * (2.0*q_err + fabs(q)*eps) + fabs(discr)*eps;

    // depending on the discriminant we get different solutions
    if(fabs(discr)<=discr_err) {
        // discriminant zero: one or two real roots
        if(fabs(p)<=p_err) {
            // p and q are zero: single root
            z.resize(1);
            z[0] = 0.0;             // triple root
        } else {
            z.resize(2);
            z[0] = 2.0*q/p;         // single root
            z[1] = -0.5*z[0];       // double root
        }
    } else if(discr>0) {
        // three real roots: via trigonometric solution
        z.resize(3);
        double ac = (1.0/3.0) * acos( q/(p*sqrt(p)) );
        double sq = 2.0*sqrt(p);
        z[0] = sq * cos(ac);
        z[1] = sq * cos(ac-2.0*M_PI/3.0);
        z[2] = sq * cos(ac-4.0*M_PI/3.0);
    } else if (discr<0.0) {
        // single real root: via Cardano's fromula
        z.resize(1);
        double sgnq = (q >= 0 ? 1 : -1);
        double basis = fabs(q) + sqrt(-discr);
        double C = sgnq * pow(basis, 1.0/3.0); // c++11 has std::cbrt()
        z[0] = C + p/C;
    }
    for(size_t i=0; i<z.size(); i++) {
        // convert depressed cubic roots to original cubic: x = z - c/3
        z[i] -= (1.0/3.0)*c;
        // improve solution via newton steps
        for(int k=0; k<newton_iter; k++) {
            double f  = ((z[i] + c)*z[i] + b)*z[i] + a;
            double f1 = (3.0*z[i] + 2.0*c)*z[i] + b;
            // only adjust if slope is large enough
            if(fabs(f1)>1e-8) {
                z[i] -= f/f1;
            }
        }
    }
    // ensure if a=0 we get exactly x=0 as root
    // TODO: remove this fudge
    if(a==0.0) {
        assert(z.size()>0);     // cubic should always have at least one root
        double xmin=fabs(z[0]);
        size_t imin=0;
        for(size_t i=1; i<z.size(); i++) {
            if(xmin>fabs(z[i])) {
                xmin=fabs(z[i]);
                imin=i;
            }
        }
        z[imin]=0.0;        // replace the smallest absolute value with 0
    }
    std::sort(z.begin(), z.end());
    return z;
}


} // namespace internal


} // namespace tk


} // namespace

#pragma GCC diagnostic pop

#endif /* TK_SPLINE_H */
// end of tk_spline.h

// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/AccelStamped.h>
#include <nav_msgs/Odometry.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>

// FS msgs
#include <fs_msgs/ControlCommand.h>
#include <fs_msgs/FinishedSignal.h>
#include <fs_msgs/GoSignal.h>

// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>

// Eigen
#include <Eigen/Dense>

// OpenCV bridge
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

// ================================================================
// ======================= CORE DATA TYPES ========================
// ================================================================

// ==================== Constants ====================
constexpr double DEG_TO_RAD = M_PI / 180.0;
constexpr double RAD_TO_DEG = 180.0 / M_PI;
constexpr double EARTH_RADIUS = 6378137.0; // WGS84 Earth radius in meters

// ==================== Data Structures ====================
struct Cone {
    pcl::PointXYZ center; // Cone center
    std::string color; // Cone color (yellow, blue, orange)
    float confidence; // Confidence level of cone detection (0.0 to 1.0)
    std::vector<pcl::PointXYZ> points; // Points belonging to the cone cluster
};

struct ColorConfidence {
    double yellow_confidence;
    double blue_confidence;
    double orange_confidence;
    double unknown_confidence;
    
    ColorConfidence() : yellow_confidence(0.0), blue_confidence(0.0), 
                       orange_confidence(0.0), unknown_confidence(1.0) {} // initialize unknown confidence to 1.0
};

struct VehicleState {
    Eigen::Vector2d position; // vehicle position in 2D (x, y)
    double yaw;               // vehicle yaw angle in radians
    double speed;             // current vehicle speed (m/s)

    // initialize with default values
    VehicleState(double x = 0.0, double y = 0.0, double y_rad = 0.0, double spd = 0.0)
        : position(x, y), yaw(y_rad), speed(spd) {}
};

struct TrajectoryPoint {
    Eigen::Vector2d position;   // x, y position
    double yaw;                 // heading angle (radians)
    double curvature;           // path curvature (1/radius)
    double speed;               // target speed (m/s)
    double s;                   // arc length from start
    
    // initialize with default values
    TrajectoryPoint(double x = 0.0, double y = 0.0, double yaw_val = 0.0, 
                   double curv = 0.0, double spd = 0.0, double s_val = 0.0)
        : position(x, y), yaw(yaw_val), curvature(curv), speed(spd), s(s_val) {}
};

// Autonomous System States according to Formula Student rules
enum class ASState {
    AS_OFF = 0,        // 자율주행 시스템이 비활성화된 상태
    AS_READY = 1,      // 주행 준비가 완료되어 오퍼레이터의 GO 신호를 대기하는 상태
    AS_DRIVING = 2,    // 자율주행으로 트랙을 주행 중인 상태
    AS_FINISHED = 3    // 레이스를 완료하고 정지를 준비하는 상태
};

// Event types that can trigger state transitions
enum class ASEvent {
    SYSTEM_INIT,        // 시스템 초기화 완료
    SYSTEM_READY,       // 모든 서브시스템 준비 완료
    GO_SIGNAL,          // 오퍼레이터 GO 신호 수신
    RACE_FINISHED       // 레이스 완료 신호
};

// State transition result
struct StateTransitionResult {
    bool success;
    ASState previous_state;
    ASState current_state;
    std::string message;
    
    // result reporting
    StateTransitionResult(bool s, ASState prev, ASState curr, const std::string& msg)
        : success(s), previous_state(prev), current_state(curr), message(msg) {}
};

// Behavior Planning
enum class DrivingMode {
    MAPPING,
    RACING
};

// =================================================================
// ==================== PARAMETER STRUCTS ==========================
// =================================================================

// Perception

struct PerceptionParams
{
    // =================== LiDAR Perception Parameters ===================
    // ROI Extraction
    double lidar_roi_x_min_;
    double lidar_roi_x_max_;
    double lidar_roi_y_min_;
    double lidar_roi_y_max_;
    double lidar_roi_z_min_;
    double lidar_roi_z_max_;

    // Ground Removal (RANSAC)
    int lidar_ransac_iterations_; 
    double lidar_ransac_distance_threshold_;
    
    // Clustering (DBSCAN)
    double lidar_dbscan_eps_; // Epsilon distance for clustering
    int lidar_dbscan_min_points_; // Minimum points to form a cluster
    
    // Cone Detection
    double lidar_cone_detection_min_height_;
    double lidar_cone_detection_max_height_;
    double lidar_cone_detection_min_radius_;
    double lidar_cone_detection_max_radius_;
    int lidar_cone_detection_min_points_;
    
    // Color Classification (removed - using camera-based classification only)
    
    // =================== Camera Parameters ===================
    
    // Camera Intrinsics
    double camera_fx_;                    // Focal length X
    double camera_fy_;                    // Focal length Y
    double camera_cx_;                    // Principal point X
    double camera_cy_;                    // Principal point Y
    
    // Camera Extrinsics (Camera coordinate system relative to vehicle base)
    std::vector<double> camera1_translation_; // [x, y, z] translation from base to camera1
    std::vector<double> camera1_rotation_;    // [roll, pitch, yaw] rotation from base to camera1 (radians)

    std::vector<double> camera2_translation_; // [x, y, z] translation from base to camera2
    std::vector<double> camera2_rotation_;    // [roll, pitch, yaw] rotation from base to camera2 (radians)
    
    // Image Processing
    bool camera_enable_preprocessing_;
    double camera_gaussian_blur_sigma_; // noise reduction sigma for Gaussian blur
    int camera_bilateral_filter_d_; // smart blur filter(maintain edges)
    
    // HSV Color Thresholds for Cone Detection
    int camera_hsv_window_size_;

    int camera_yellow_hue_min_;
    int camera_yellow_hue_max_;
    int camera_yellow_sat_min_;
    int camera_yellow_val_min_;
    
    int camera_blue_hue_min_;
    int camera_blue_hue_max_;
    int camera_blue_sat_min_;
    int camera_blue_val_min_;
    
    int camera_orange_hue_min_;
    int camera_orange_hue_max_;
    int camera_orange_sat_min_;
    int camera_orange_val_min_;
    
    // =================== LiDAR Extrinsics ===================
    
    // LiDAR Extrinsics (LiDAR coordinate system relative to vehicle base)
    std::vector<double> lidar_translation_; // [x, y, z] translation from base to lidar
    std::vector<double> lidar_rotation_;    // [roll, pitch, yaw] rotation from base to lidar (radians)

    // Print parameters for debugging
    void print() const {
        printf("=== Perception Parameters ===\n");
        printf("\n[LiDAR Parameters]\n");
        printf("  ROI Extraction: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n", lidar_roi_x_min_, lidar_roi_x_max_, lidar_roi_y_min_, lidar_roi_y_max_, lidar_roi_z_min_, lidar_roi_z_max_);
        printf("  RANSAC iterations: %d\n", lidar_ransac_iterations_);
        printf("  RANSAC distance threshold: %.3f\n", lidar_ransac_distance_threshold_);
        printf("  DBSCAN eps: %.3f\n", lidar_dbscan_eps_);
        printf("  DBSCAN min points: %d\n", lidar_dbscan_min_points_);
        printf("  Cone height range: [%.3f, %.3f]\n", lidar_cone_detection_min_height_, lidar_cone_detection_max_height_);
        printf("  Cone radius range: [%.3f, %.3f]\n", lidar_cone_detection_min_radius_, lidar_cone_detection_max_radius_);
        printf("  Cone min points: %d\n", lidar_cone_detection_min_points_);
        printf("  Translation: [%.3f, %.3f, %.3f]\n", lidar_translation_[0], lidar_translation_[1], lidar_translation_[2]);
        printf("  Rotation: [%.3f, %.3f, %.3f] deg\n", lidar_rotation_[0], lidar_rotation_[1], lidar_rotation_[2]);
        
        printf("\n[Camera Parameters]\n");
        printf("  Intrinsics: fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f\n", camera_fx_, camera_fy_, camera_cx_, camera_cy_);
        printf("  Camera 1 (Left) Translation: [%.3f, %.3f, %.3f]\n", camera1_translation_[0], camera1_translation_[1], camera1_translation_[2]);
        printf("  Camera 1 (Left) Rotation: [%.3f, %.3f, %.3f] deg\n", camera1_rotation_[0], camera1_rotation_[1], camera1_rotation_[2]);
        printf("  Camera 2 (Right) Translation: [%.3f, %.3f, %.3f]\n", camera2_translation_[0], camera2_translation_[1], camera2_translation_[2]);
        printf("  Camera 2 (Right) Rotation: [%.3f, %.3f, %.3f] deg\n", camera2_rotation_[0], camera2_rotation_[1], camera2_rotation_[2]);
        printf("  Preprocessing enabled: %s\n", camera_enable_preprocessing_ ? "true" : "false");
        printf("  Gaussian blur sigma: %.3f\n", camera_gaussian_blur_sigma_);
        printf("  Bilateral filter diameter: %d\n", camera_bilateral_filter_d_);
        printf("  HSV window size: %d\n", camera_hsv_window_size_);
        printf("  Yellow HSV range: H[%d,%d] S[%d,255] V[%d,255]\n", 
               camera_yellow_hue_min_, camera_yellow_hue_max_, camera_yellow_sat_min_, camera_yellow_val_min_);
        printf("  Blue HSV range: H[%d,%d] S[%d,255] V[%d,255]\n", 
               camera_blue_hue_min_, camera_blue_hue_max_, camera_blue_sat_min_, camera_blue_val_min_);
        printf("  Orange HSV range: H[%d,%d] S[%d,255] V[%d,255]\n", 
               camera_orange_hue_min_, camera_orange_hue_max_, camera_orange_sat_min_, camera_orange_val_min_);
    }
    
    bool getParameters(ros::NodeHandle& pnh){
        std::cout << "FormulaAutonomousSystem: Parameters file updated" << std::endl;
        // =================== LiDAR Parameters ===================
        // LiDAR ROI Extraction
        if(!pnh.getParam("/perception/lidar_roi_extraction/x_min", lidar_roi_x_min_)){std::cerr<<"Param perception/lidar_roi_extraction/x_min has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/lidar_roi_extraction/x_max", lidar_roi_x_max_)){std::cerr<<"Param perception/lidar_roi_extraction/x_max has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/lidar_roi_extraction/y_min", lidar_roi_y_min_)){std::cerr<<"Param perception/lidar_roi_extraction/y_min has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/lidar_roi_extraction/y_max", lidar_roi_y_max_)){std::cerr<<"Param perception/lidar_roi_extraction/y_max has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/lidar_roi_extraction/z_min", lidar_roi_z_min_)){std::cerr<<"Param perception/lidar_roi_extraction/z_min has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/lidar_roi_extraction/z_max", lidar_roi_z_max_)){std::cerr<<"Param perception/lidar_roi_extraction/z_max has error" << std::endl; return false;}

        // LiDAR Ground Removal
        if(!pnh.getParam("/perception/lidar_ground_removal/ransac_iterations", lidar_ransac_iterations_)){std::cerr<<"Param perception/lidar_ground_removal/ransac_iterations has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/lidar_ground_removal/ransac_distance_threshold", lidar_ransac_distance_threshold_)){std::cerr<<"Param perception/lidar_ground_removal/ransac_distance_threshold has error" << std::endl; return false;}

        // LiDAR Clustering
        if(!pnh.getParam("/perception/lidar_clustering/dbscan_eps", lidar_dbscan_eps_)){std::cerr<<"Param perception/lidar_clustering/dbscan_eps has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/lidar_clustering/dbscan_min_points", lidar_dbscan_min_points_)){std::cerr<<"Param perception/lidar_clustering/dbscan_min_points has error" << std::endl; return false;}

        // LiDAR Cone Detection
        if(!pnh.getParam("/perception/lidar_cone_detection/cone_min_height", lidar_cone_detection_min_height_)){std::cerr<<"Param perception/lidar_cone_detection/cone_min_height has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/lidar_cone_detection/cone_max_height", lidar_cone_detection_max_height_)){std::cerr<<"Param perception/lidar_cone_detection/cone_max_height has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/lidar_cone_detection/cone_min_radius", lidar_cone_detection_min_radius_)){std::cerr<<"Param perception/lidar_cone_detection/cone_min_radius has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/lidar_cone_detection/cone_max_radius", lidar_cone_detection_max_radius_)){std::cerr<<"Param perception/lidar_cone_detection/cone_max_radius has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/lidar_cone_detection/cone_min_points", lidar_cone_detection_min_points_)){std::cerr<<"Param perception/lidar_cone_detection/cone_min_points has error" << std::endl; return false;}

        // LiDAR Extrinsics
        if(lidar_translation_.size() != 3) lidar_translation_.resize(3);
        if(lidar_rotation_.size() != 3) lidar_rotation_.resize(3);
        
        if(!pnh.getParam("/perception/lidar_extrinsics/translation_x", lidar_translation_[0])){std::cerr<<"Param perception/lidar_extrinsics/translation_x has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/lidar_extrinsics/translation_y", lidar_translation_[1])){std::cerr<<"Param perception/lidar_extrinsics/translation_y has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/lidar_extrinsics/translation_z", lidar_translation_[2])){std::cerr<<"Param perception/lidar_extrinsics/translation_z has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/lidar_extrinsics/rotation_roll", lidar_rotation_[0])){std::cerr<<"Param perception/lidar_extrinsics/rotation_roll has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/lidar_extrinsics/rotation_pitch", lidar_rotation_[1])){std::cerr<<"Param perception/lidar_extrinsics/rotation_pitch has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/lidar_extrinsics/rotation_yaw", lidar_rotation_[2])){std::cerr<<"Param perception/lidar_extrinsics/rotation_yaw has error" << std::endl; return false;}

        // =================== Camera Parameters ===================
        // Camera Intrinsics
        if(!pnh.getParam("/perception/camera_intrinsics/focal_length_x", camera_fx_)){std::cerr<<"Param perception/camera_intrinsics/focal_length_x has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera_intrinsics/focal_length_y", camera_fy_)){std::cerr<<"Param perception/camera_intrinsics/focal_length_y has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera_intrinsics/principal_point_x", camera_cx_)){std::cerr<<"Param perception/camera_intrinsics/principal_point_x has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera_intrinsics/principal_point_y", camera_cy_)){std::cerr<<"Param perception/camera_intrinsics/principal_point_y has error" << std::endl; return false;}

        // Camera1 Extrinsics
        if(camera1_translation_.size() != 3) camera1_translation_.resize(3);
        if(camera1_rotation_.size() != 3) camera1_rotation_.resize(3);
        
        if(!pnh.getParam("/perception/camera1_extrinsics/translation_x", camera1_translation_[0])){std::cerr<<"Param perception/camera1_extrinsics/translation_x has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera1_extrinsics/translation_y", camera1_translation_[1])){std::cerr<<"Param perception/camera1_extrinsics/translation_y has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera1_extrinsics/translation_z", camera1_translation_[2])){std::cerr<<"Param perception/camera1_extrinsics/translation_z has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera1_extrinsics/rotation_roll", camera1_rotation_[0])){std::cerr<<"Param perception/camera1_extrinsics/rotation_roll has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera1_extrinsics/rotation_pitch", camera1_rotation_[1])){std::cerr<<"Param perception/camera1_extrinsics/rotation_pitch has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera1_extrinsics/rotation_yaw", camera1_rotation_[2])){std::cerr<<"Param perception/camera1_extrinsics/rotation_yaw has error" << std::endl; return false;}

        // Camera2 Extrinsics
        if(camera2_translation_.size() != 3) camera2_translation_.resize(3);
        if(camera2_rotation_.size() != 3) camera2_rotation_.resize(3);

        if(!pnh.getParam("/perception/camera2_extrinsics/translation_x", camera2_translation_[0])){std::cerr<<"Param perception/camera2_extrinsics/translation_x has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera2_extrinsics/translation_y", camera2_translation_[1])){std::cerr<<"Param perception/camera2_extrinsics/translation_y has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera2_extrinsics/translation_z", camera2_translation_[2])){std::cerr<<"Param perception/camera2_extrinsics/translation_z has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera2_extrinsics/rotation_roll", camera2_rotation_[0])){std::cerr<<"Param perception/camera2_extrinsics/rotation_roll has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera2_extrinsics/rotation_pitch", camera2_rotation_[1])){std::cerr<<"Param perception/camera2_extrinsics/rotation_pitch has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera2_extrinsics/rotation_yaw", camera2_rotation_[2])){std::cerr<<"Param perception/camera2_extrinsics/rotation_yaw has error" << std::endl; return false;}

        // Camera Image Processing
        if(!pnh.getParam("/perception/camera_image_processing/enable_preprocessing", camera_enable_preprocessing_)){std::cerr<<"Param perception/camera_image_processing/enable_preprocessing has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera_image_processing/gaussian_blur_sigma", camera_gaussian_blur_sigma_)){std::cerr<<"Param perception/camera_image_processing/gaussian_blur_sigma has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera_image_processing/bilateral_filter_diameter", camera_bilateral_filter_d_)){std::cerr<<"Param perception/camera_image_processing/bilateral_filter_diameter has error" << std::endl; return false;}

        // Camera HSV Window Size
        if(!pnh.getParam("/perception/camera_hsv_window_size/window_size", camera_hsv_window_size_)){std::cerr<<"Param perception/camera_hsv_window_size/window_size has error" << std::endl; return false;}

        // Camera HSV Yellow
        if(!pnh.getParam("/perception/camera_hsv_yellow/hue_min", camera_yellow_hue_min_)){std::cerr<<"Param perception/camera_hsv_yellow/hue_min has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera_hsv_yellow/hue_max", camera_yellow_hue_max_)){std::cerr<<"Param perception/camera_hsv_yellow/hue_max has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera_hsv_yellow/saturation_min", camera_yellow_sat_min_)){std::cerr<<"Param perception/camera_hsv_yellow/saturation_min has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera_hsv_yellow/value_min", camera_yellow_val_min_)){std::cerr<<"Param perception/camera_hsv_yellow/value_min has error" << std::endl; return false;}

        // Camera HSV Blue
        if(!pnh.getParam("/perception/camera_hsv_blue/hue_min", camera_blue_hue_min_)){std::cerr<<"Param perception/camera_hsv_blue/hue_min has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera_hsv_blue/hue_max", camera_blue_hue_max_)){std::cerr<<"Param perception/camera_hsv_blue/hue_max has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera_hsv_blue/saturation_min", camera_blue_sat_min_)){std::cerr<<"Param perception/camera_hsv_blue/saturation_min has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera_hsv_blue/value_min", camera_blue_val_min_)){std::cerr<<"Param perception/camera_hsv_blue/value_min has error" << std::endl; return false;}

        // Camera HSV Orange
        if(!pnh.getParam("/perception/camera_hsv_orange/hue_min", camera_orange_hue_min_)){std::cerr<<"Param perception/camera_hsv_orange/hue_min has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera_hsv_orange/hue_max", camera_orange_hue_max_)){std::cerr<<"Param perception/camera_hsv_orange/hue_max has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera_hsv_orange/saturation_min", camera_orange_sat_min_)){std::cerr<<"Param perception/camera_hsv_orange/saturation_min has error" << std::endl; return false;}
        if(!pnh.getParam("/perception/camera_hsv_orange/value_min", camera_orange_val_min_)){std::cerr<<"Param perception/camera_hsv_orange/value_min has error" << std::endl; return false;}

        return true;
    }
};

// ==================== Localization ====================

struct LocalizationParams
{
    // =================== Reference WGS84 Position ===================
    bool use_user_defined_ref_wgs84_position_;
    double ref_wgs84_latitude_;      // Reference latitude (degrees)
    double ref_wgs84_longitude_;     // Reference longitude (degrees)
    double ref_wgs84_altitude_;      // Reference altitude (meters)

    // =================== Velocity Estimation ===================
    double gps_correction_gain_;

    // Print parameters for debugging
    void print() const {
        printf("=== Localization Parameters ===\n");
        printf("\n[Reference WGS84 Position]\n");
        printf("  Use user defined: %s\n", use_user_defined_ref_wgs84_position_ ? "true" : "false");
        printf("  Latitude: %.6f°\n", ref_wgs84_latitude_);
        printf("  Longitude: %.6f°\n", ref_wgs84_longitude_);
        printf("  Altitude: %.3fm\n", ref_wgs84_altitude_);

        printf("\n[Velocity Estimation]\n");
        printf("  GPS correction gain: %.6f\n", gps_correction_gain_);
    }

    bool getParameters(ros::NodeHandle& pnh){
        std::cout << "FormulaAutonomousSystem: Localization parameters file updated" << std::endl;
        
        // =================== Localization Parameters ===================
        if(!pnh.getParam("/localization/use_user_defined_ref_wgs84_position", use_user_defined_ref_wgs84_position_)){std::cerr<<"Param localization/use_user_defined_ref_wgs84_position has error" << std::endl; return false;}
        if(!pnh.getParam("/localization/ref_wgs84_latitude", ref_wgs84_latitude_)){std::cerr<<"Param localization/ref_wgs84_latitude has error" << std::endl; return false;}
        if(!pnh.getParam("/localization/ref_wgs84_longitude", ref_wgs84_longitude_)){std::cerr<<"Param localization/ref_wgs84_longitude has error" << std::endl; return false;}
        if(!pnh.getParam("/localization/ref_wgs84_altitude", ref_wgs84_altitude_)){std::cerr<<"Param localization/ref_wgs84_altitude has error" << std::endl; return false;}
        if(!pnh.getParam("/localization/gps_correction_gain", gps_correction_gain_)){std::cerr<<"Param localization/gps_correction_gain has error" << std::endl; return false;}

        return true;
    }
};

// ===================== Mapping =====================

struct MappingParams {
    // =================== Mapping Parameters ===================
    double cone_memory_search_radius_;              // Search radius for associating new cones with memory (m)
    double cone_memory_association_threshold_;      // Minimum association confidence to consider a match
    double max_connection_distance_;                // Maximum distance to connect cones (m)
    double direction_weight_;                       // Weight for direction consistency in cone sorting
    double max_dist_from_lane_;                     // Maximum distance from lane to consider a cone valid (m)
    
    // Print parameters for debugging
    void print() const {
        printf("=== Mapping Parameters ===\n");
        printf("Cone memory search radius: %.3f m\n", cone_memory_search_radius_);
        printf("Cone memory association threshold: %.3f m\n", cone_memory_association_threshold_);
        printf("Max connection distance: %.3f m\n", max_connection_distance_);
        printf("Direction wight: %.3f m\n", direction_weight_);
        printf("Max dist from lane: %.3f m\n", max_dist_from_lane_);
    }
    
    // Load parameters from ROS NodeHandle
    bool getParameters(ros::NodeHandle& pnh) {
        std::cout << "FormulaAutonomousSystem: Mapping parameters file updated" << std::endl;
        
        // =================== Mapping Parameters ===================
        if(!pnh.getParam("/mapping/cone_memory_search_radius", cone_memory_search_radius_)){std::cerr<<"Param mapping/cone_memory_search_radius has error" << std::endl; return false;}
        if(!pnh.getParam("/mapping/cone_memory_association_threshold", cone_memory_association_threshold_)){std::cerr<<"Param mapping/cone_memory_association_threshold has error" << std::endl; return false;}
        if(!pnh.getParam("/mapping/max_connection_distance", max_connection_distance_)){std::cerr<<"Param mapping/max_connection_distance has error" << std::endl; return false;}
        if(!pnh.getParam("/mapping/direction_weight", direction_weight_)){std::cerr<<"Param mapping/direction_weight has error" << std::endl; return false;}
        if(!pnh.getParam("/mapping/max_dist_from_lane", max_dist_from_lane_)){std::cerr<<"Param mapping/max_dist_from_lane has error" << std::endl; return false;}
        
        return true;
    }
};

// ==================== Planning ====================

struct PlanningParams {

    // Structure to hold parameters for a specific trajectory generation mode
    struct TrajectoryModeParams {
        double lookahead_distance_;         // Lookahead distance (m)
        double waypoint_spacing_;           // Distance between waypoints (m)
        double max_speed_;                  // Max speed (m/s)
        double min_speed_;                  // Minimum speed (m/s)
        double curvature_gain_;             // curvature gain for speed adjustment
        double curvature_threshold_;        // threshold for determining critical section
        double min_segment_length_;         // critical section length (m)
        double lane_offset_;                // Offset from single cone (m)
    };

    struct TrajectoryGeneration {
        TrajectoryModeParams mapping_mode;
        TrajectoryModeParams racing_mode;
    };

    struct BehavioralLogic {
        int total_laps_;
        double finish_stop_distance_;
    };
    
    // Main parameter holders
    TrajectoryGeneration trajectory_generation;
    BehavioralLogic behavioral_logic;

    // Print current parameters
    void print() const {
        printf("=== Planning Parameters ===\n");
        
        printf("\n[Trajectory Generation -> Mapping Mode]\n");
        printf("  Lookahead Distance: %.3f m\n", trajectory_generation.mapping_mode.lookahead_distance_);
        printf("  Waypoint Spacing: %.3f m\n", trajectory_generation.mapping_mode.waypoint_spacing_);
        printf("  Max Speed: %.3f m/s\n", trajectory_generation.mapping_mode.max_speed_);
        printf("  Lane Offset: %.3f m\n", trajectory_generation.mapping_mode.lane_offset_);

        printf("\n[Trajectory Generation -> Racing Mode]\n");
        printf("  Lookahead Distance: %.3f m\n", trajectory_generation.racing_mode.lookahead_distance_);
        printf("  Waypoint Spacing: %.3f m\n", trajectory_generation.racing_mode.waypoint_spacing_);
        printf("  Max Speed: %.3f m/s\n", trajectory_generation.racing_mode.max_speed_);
        printf("  Min Speed: %.3f m/s\n", trajectory_generation.racing_mode.min_speed_);
        printf("  Curvature Gain: %.3f\n", trajectory_generation.racing_mode.curvature_gain_);
        printf("  Curvature Threshold: %.3f\n", trajectory_generation.racing_mode.curvature_threshold_);
        printf("  Min Segment Length: %.3f m\n", trajectory_generation.racing_mode.min_segment_length_);

        printf("\n[Behavioral Logic]\n");
        printf("  Total Laps: %d\n", behavioral_logic.total_laps_);
        printf("  Finish Stop Distance: %.3f m\n", behavioral_logic.finish_stop_distance_);
    }

    // Load all planning parameters from ROS NodeHandle
    bool getParameters(ros::NodeHandle& pnh) {
        std::cout << "FormulaAutonomousSystem: Loading Planning parameters..." << std::endl;
        
        // Load Trajectory Generation -> Mapping Mode 
        if(!pnh.getParam("/planning/trajectory_generation/mapping_mode/lookahead_distance", trajectory_generation.mapping_mode.lookahead_distance_)){std::cerr<<"Param /planning/trajectory_generation/mapping_mode/lookahead_distance" << std::endl; return false; }
        if(!pnh.getParam("/planning/trajectory_generation/mapping_mode/waypoint_spacing", trajectory_generation.mapping_mode.waypoint_spacing_)){std::cerr<<"Param /planning/trajectory_generation/mapping_mode/waypoint_spacing" << std::endl; return false; }
        if(!pnh.getParam("/planning/trajectory_generation/mapping_mode/max_speed", trajectory_generation.mapping_mode.max_speed_)){std::cerr<<"Param /planning/trajectory_generation/mapping_mode/max_speed" << std::endl; return false; }
        if(!pnh.getParam("/planning/trajectory_generation/mapping_mode/lane_offset", trajectory_generation.mapping_mode.lane_offset_)){std::cerr<<"Param /planning/trajectory_generation/mapping_mode/lane_offset" << std::endl; return false; }

        // Load Trajectory Generation -> Racing Mode parameters
        if(!pnh.getParam("/planning/trajectory_generation/racing_mode/lookahead_distance", trajectory_generation.racing_mode.lookahead_distance_)){std::cerr<<"Param /planning/trajectory_generation/racing_mode/lookahead_distance" << std::endl; return false; }
        if(!pnh.getParam("/planning/trajectory_generation/racing_mode/waypoint_spacing", trajectory_generation.racing_mode.waypoint_spacing_)){std::cerr<<"Param /planning/trajectory_generation/racing_mode/lookahead_distance" << std::endl; return false; }
        if(!pnh.getParam("/planning/trajectory_generation/racing_mode/max_speed", trajectory_generation.racing_mode.max_speed_)){std::cerr<<"Param /planning/trajectory_generation/racing_mode/lookahead_distance" << std::endl; return false; }
        if(!pnh.getParam("/planning/trajectory_generation/racing_mode/min_speed", trajectory_generation.racing_mode.min_speed_)){std::cerr<<"Param /planning/trajectory_generation/racing_mode/lookahead_distance" << std::endl; return false; }
        if(!pnh.getParam("/planning/trajectory_generation/racing_mode/curvature_gain", trajectory_generation.racing_mode.curvature_gain_)){std::cerr<<"Param /planning/trajectory_generation/racing_mode/lookahead_distance" << std::endl; return false; }
        if(!pnh.getParam("/planning/trajectory_generation/racing_mode/curvature_threshold", trajectory_generation.racing_mode.curvature_threshold_)){std::cerr<<"Param /planning/trajectory_generation/racing_mode/curvature_threshold" << std::endl; return false; }
        if(!pnh.getParam("/planning/trajectory_generation/racing_mode/min_segment_length", trajectory_generation.racing_mode.min_segment_length_)){std::cerr<<"Param /planning/trajectory_generation/racing_mode/min_segment_length" << std::endl; return false; }

        // Load Behavioral Logic parameters
        if(!pnh.getParam("/planning/behavioral_logic/total_laps", behavioral_logic.total_laps_)){std::cerr<<"Param /planning/behavioral_logic/total_laps" << std::endl; return false; }
        if(!pnh.getParam("/planning/behavioral_logic/finish_stop_distance", behavioral_logic.finish_stop_distance_)){std::cerr<<"Param /planning/behavioral_logic/finish_stop_distance" << std::endl; return false; } // 추가
        
        return true;
    }
};

// ==================== Control ====================

struct ControlParams {
    // Structure to hold parameters for a specific controller mode
    struct ControllerModeParams {
        // Lateral Control: Pure Pursuit
        double pp_lookahead_distance_;
        double pp_max_steer_angle_;

        // Lateral Control: Stanley
        double k_gain_;
        double k_gain_curvature_boost_; // Note: This is only used in racing_mode
        double stanley_alpha_;

        // Longitudinal Control: PID Controller
        double pid_kp_;
        double pid_ki_;
        double pid_kd_;
        double max_throttle_;
        double steering_based_speed_gain_;
    };

    // ===================  Controller Selection ===================
    std::string lateral_controller_type_;

    // Mode-specific parameter sets
    ControllerModeParams mapping_mode;
    ControllerModeParams racing_mode;

    // =================== Vehicle Specification ===================
    double vehicle_length_;

    bool getParameters(ros::NodeHandle& pnh) {
        std::cout << "FormulaAutonomousSystem: Control parameters file updated" << std::endl;
        
        // ===================  Controller Selection ===================
        if(!pnh.getParam("/control/ControllerSelection/lateral_controller_type", lateral_controller_type_)){std::cerr<<"Param control/ControllerSelection/lateral_controller_type has error" << std::endl; return false;}

        // ===================  Mapping Mode Parameters ===================
        // Pure Pursuit
        if(!pnh.getParam("/control/mapping_mode/PurePursuit/lookahead_distance", mapping_mode.pp_lookahead_distance_)){std::cerr<<"Param control/mapping_mode/PurePursuit/lookahead_distance has error" << std::endl; return false;}
        if(!pnh.getParam("/control/mapping_mode/PurePursuit/max_steer_angle", mapping_mode.pp_max_steer_angle_)){std::cerr<<"Param control/mapping_mode/PurePursuit/max_steer_angle has error" << std::endl; return false;}

        // Stanley
        if(!pnh.getParam("/control/mapping_mode/Stanley/k_gain", mapping_mode.k_gain_)){std::cerr<<"Param control/mapping_mode/Stanley/k_gain has error" << std::endl; return false;}
        if(!pnh.getParam("/control/mapping_mode/Stanley/alpha", mapping_mode.stanley_alpha_)){std::cerr<<"Param control/mapping_mode/Stanley/alpha has error" << std::endl; return false;}
        mapping_mode.k_gain_curvature_boost_ = 0.0; // Noit used in mapping mode, reset to 0.0

        // SpeedControl
        if(!pnh.getParam("/control/mapping_mode/SpeedControl/pid_kp", mapping_mode.pid_kp_)){std::cerr<<"Param control/mapping_mode/SpeedControl/pid_kp has error" << std::endl; return false;}
        if(!pnh.getParam("/control/mapping_mode/SpeedControl/pid_ki", mapping_mode.pid_ki_)){std::cerr<<"Param control/mapping_mode/SpeedControl/pid_ki has error" << std::endl; return false;}
        if(!pnh.getParam("/control/mapping_mode/SpeedControl/pid_kd", mapping_mode.pid_kd_)){std::cerr<<"Param control/mapping_mode/SpeedControl/pid_kd has error" << std::endl; return false;}
        if(!pnh.getParam("/control/mapping_mode/SpeedControl/max_throttle", mapping_mode.max_throttle_)){std::cerr<<"Param control/mapping_mode/SpeedControl/max_throttle has error" << std::endl; return false;}
        if(!pnh.getParam("/control/mapping_mode/SpeedControl/steering_based_speed_gain", mapping_mode.steering_based_speed_gain_)){std::cerr<<"Param control/mapping_mode/SpeedControl/steering_based_speed_gain has error" << std::endl; return false;}

        // ===================  Racing Mode Parameters ===================
        // Pure Pursuit
        if(!pnh.getParam("/control/racing_mode/PurePursuit/lookahead_distance", racing_mode.pp_lookahead_distance_)){std::cerr<<"Param control/racing_mode/PurePursuit/lookahead_distance has error" << std::endl; return false;}
        if(!pnh.getParam("/control/racing_mode/PurePursuit/max_steer_angle", racing_mode.pp_max_steer_angle_)){std::cerr<<"Param control/racing_mode/PurePursuit/max_steer_angle has error" << std::endl; return false;}

        // Stanley
        if(!pnh.getParam("/control/racing_mode/Stanley/k_gain", racing_mode.k_gain_)){std::cerr<<"Param control/racing_mode/Stanley/k_gain has error" << std::endl; return false;}
        if(!pnh.getParam("/control/racing_mode/Stanley/k_gain_curvature_boost", racing_mode.k_gain_curvature_boost_)){std::cerr<<"Param control/racing_mode/Stanley/k_gain_curvature_boost has error" << std::endl; return false;}
        if(!pnh.getParam("/control/racing_mode/Stanley/alpha", racing_mode.stanley_alpha_)){std::cerr<<"Param control/racing_mode/Stanley/alpha has error" << std::endl; return false;}

        // SpeedControl
        if(!pnh.getParam("/control/racing_mode/SpeedControl/pid_kp", racing_mode.pid_kp_)){std::cerr<<"Param control/racing_mode/SpeedControl/pid_kp has error" << std::endl; return false;}
        if(!pnh.getParam("/control/racing_mode/SpeedControl/pid_ki", racing_mode.pid_ki_)){std::cerr<<"Param control/racing_mode/SpeedControl/pid_ki has error" << std::endl; return false;}
        if(!pnh.getParam("/control/racing_mode/SpeedControl/pid_kd", racing_mode.pid_kd_)){std::cerr<<"Param control/racing_mode/SpeedControl/pid_kd has error" << std::endl; return false;}
        if(!pnh.getParam("/control/racing_mode/SpeedControl/max_throttle", racing_mode.max_throttle_)){std::cerr<<"Param control/racing_mode/SpeedControl/max_throttle has error" << std::endl; return false;}
        if(!pnh.getParam("/control/racing_mode/SpeedControl/steering_based_speed_gain", racing_mode.steering_based_speed_gain_)){std::cerr<<"Param control/racing_mode/SpeedControl/steering_based_speed_gain has error" << std::endl; return false;}

        // =================== Vehicle Specification ===================
        if(!pnh.getParam("/control/Vehicle/wheel_base", vehicle_length_)){std::cerr<<"Param control/Vehicle/wheel_base has error" << std::endl; return false;}

        return true;
    }
};

// ====================================================================
// ======================== ALGORITHM CLASSES =========================
// ====================================================================

// Perception

class RoiExtractor {
public:
    RoiExtractor(const std::shared_ptr<PerceptionParams> params);
    
    // Extracts the roi_cloud from the input_cloud based on the params_
    void extractRoi(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& roi_cloud);

private:
    // Parameters for ROI extraction
    std::shared_ptr<PerceptionParams> params_;
};

class GroundRemoval {
public:
    // Constructor with parameter struct
    explicit GroundRemoval(const std::shared_ptr<PerceptionParams> params);
    
    // Legacy constructor for backward compatibility
    GroundRemoval(double distance_threshold = 0.2, int max_iterations = 1000);
    
    // Extracts ground points and non-ground points from the input cloud
    void removeGround(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& ground_points,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& non_ground_points
    );

private:
    std::shared_ptr<PerceptionParams> params_;
    
    // RANSAC 3-point plane fitting once
    bool fitPlane(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        Eigen::Vector4f& plane_coefficients,
        std::vector<int>& inliers
    );
    
    // Computes the distance from a point to a plane defined by its coefficients
    double pointToPlaneDistance(const pcl::PointXYZ& point, const Eigen::Vector4f& plane);
    std::mt19937 rng_;
};

class Clustering {
public:
    // Constructor with parameter struct
    explicit Clustering(const std::shared_ptr<PerceptionParams> params);
    
    // Legacy constructor for backward compatibility
    Clustering(double eps = 0.5, int min_points = 10, double min_cone_height = 0.1, double max_cone_height = 0.4);
    
    bool extractCones(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_points, std::vector<Cone>& cones);

private:
    std::shared_ptr<PerceptionParams> params_;
    Eigen::Matrix4f vehicle_to_lidar_transform_;
    
    // DBSCAN clustering
    std::vector<std::vector<int>> dbscan(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    
    std::vector<int> regionQuery(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        int point_idx,
        const pcl::search::KdTree<pcl::PointXYZ>::Ptr& kdtree // search engine
    );
    
    // Cone validation and color classification
    bool isValidCone(const std::vector<int>& cluster_indices, 
                        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

    Eigen::Vector3f calculateCentroid(const std::vector<int>& cluster_indices,
                                        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
};


class ColorDetection
{
public:
    explicit ColorDetection(const std::shared_ptr<PerceptionParams>& params);
    
    cv::Mat ConesColor(std::vector<Cone>& cones, sensor_msgs::Image& camera1_msg, sensor_msgs::Image& camera2_msg);
    
    // Utility functions
    cv::Point2f projectToCamera(const pcl::PointXYZ& point_3d, int camera_id); // which camera to project to
    bool isPointInImage(const cv::Point2f& point, const cv::Size& image_size);
    
    // Debug visualization
    cv::Mat visualizeProjection(const std::vector<Cone>& cones, const cv::Mat& rgb_image);
    
    // Get computed transformation matrices (for debugging)
    cv::Mat getCameraToBaseRotation() const { return camera1_to_base_rotation_.clone(); }
    cv::Mat getCameraToBaseTranslation() const { return camera1_to_base_translation_.clone(); }
    cv::Mat getCamera2ToBaseRotation() const { return camera2_to_base_rotation_.clone(); }
    cv::Mat getCamera2ToBaseTranslation() const { return camera2_to_base_translation_.clone(); }

private:
    std::shared_ptr<PerceptionParams> params_;

    // Camera intrinsic matrix
    cv::Mat camera_matrix_;
    
    // Computed transformation from camera to vehicle coordinate system
    cv::Mat camera1_to_base_rotation_;    // camera 1 (left) 3x3 rotation matrix
    cv::Mat camera1_to_base_translation_; // camera 1 (left) 3x1 translation vector
    cv::Mat camera2_to_base_rotation_;    // camera 2 (right) 3x3 rotation matrix
    cv::Mat camera2_to_base_translation_; // camera 2 (right) 3x1 translation vector
    
    // Initialize camera parameters and compute transformations
    void initializeCameraParameters();
    void computeCameraToLidarTransform();
    
    // Helper function to create rotation matrix from euler angles
    cv::Mat createTransformationMatrix(double x, double y, double z, double roll, double pitch, double yaw);

    // Convert ROS image message to OpenCV Mat
    void getCameraImage(sensor_msgs::Image& msg, cv::Mat& image);

    // Helper function to detect cone color
    std::string detectConeColor(const Cone& cone, const cv::Mat& rgb_image, const cv::Point2f& projection_point);
    
    // Color analysis functions
    ColorConfidence analyzeColorWindow(const cv::Mat& hsv_image, const cv::Point2f& center, int window_size);
    
    int countYellowPixels(const cv::Mat& hsv_roi);
    int countBluePixels(const cv::Mat& hsv_roi);
    int countOrangePixels(const cv::Mat& hsv_roi);
    
    std::string selectBestColor(const ColorConfidence& confidence);
    
    // Image preprocessing
    cv::Mat preprocessImage(const cv::Mat& rgb_image);
    
    // Helper functions
    bool isInHSVRange(const cv::Vec3b& hsv_pixel, int hue_min, int hue_max, int sat_min, int val_min);
    cv::Rect getSafeWindow(const cv::Point2f& center, int window_size, const cv::Size& image_size);

    // Main color detection function
    std::vector<Cone> classifyConesColor(const std::vector<Cone>& cones, const cv::Mat& rgb_image1, const cv::Mat& rgb_image2); // rgb_image1: camera 1 (left), rgb_image2: camera 2 (right)
};

// Localization

class Localization {
public:
    explicit Localization(const std::shared_ptr<LocalizationParams>& params);
    ~Localization() = default;

public:
    // Update with IMU data (acceleration, yaw rate and IMU orientation)
    void updateImu(const Eigen::Vector3d& imu_input, Eigen::Quaterniond& q_imu, double curr_time_sec);
    
    // Update with GPS data (position in WGS84)
    void updateGps(const Eigen::Vector2d& gps_wgs84, double curr_time_sec);
    
    // Getters
    std::vector<double> getCurrentState() const { return state_; }
    Eigen::Vector3d getCurrentPose() const { return Eigen::Vector3d(state_[0], state_[1], state_[2]); }
    double getCurrentVelocity() const { return state_[3]; }
    double getCurrentYaw() const { return state_[2]; }
    double getCurrentYawRate() const { return state_[5]; }
    double getCurrentAcceleration() const { return state_[6]; }
    double getCurrentLateralAcceleration() const { return state_[7]; }
    
    // Transform utilities
    Eigen::Vector2d wgs84ToEnu(const Eigen::Vector2d& wgs84_pos) const;

private:
    std::vector<double> predictState(const std::vector<double>& state, double dt);

private:
    std::shared_ptr<LocalizationParams> params_;

    std::vector<double> state_; // [x, y, yaw, vx, vy, yawrate, ax, ay]
    double last_time_sec_;

    Eigen::Vector2d ref_wgs84_position_; // [lat, lon]
    Eigen::Vector2d prev_gps_enu_; // [x, y]
    double prev_gps_time_sec_;
};

// Mapping
/**
 * @class MapManager
 * @brief Updates and manages the global cone map
 */
class MapManager {
public:
    explicit MapManager(const std::shared_ptr<MappingParams>& params);

    /**
     * @brief Updates the global cone map with new cones and returns the cone list to be used for planning
     * @param current_state vehicle state (global frame)
     * @param real_time_cones real-time detected cones (vehicle frame)
     * @return std::vector<Cone> Updated list of cones for planning (vehicle frame)
     */

    std::vector<Cone> updateAndGetPlannedCones(const VehicleState& current_state, const std::vector<Cone>& real_time_cones);
    std::vector<Cone> getGlobalConeMap() const;
    void generateLanesFromMemory();
    std::pair<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>> getTrackLanes();
    void refineConeMap();

private:
    // Function
    std::vector<Eigen::Vector2d> sortConesByProximity(const std::vector<Eigen::Vector2d>& cones);

    /**
     * @brief Helper function to calculate the shortest distance between a point and a line segment.
     * @param p The target point.
     * @param v The starting point of the line segment.
     * @param w The end point of the line segment.
     * @return double The shortest distance between the point and the line segment.
     */

    double pointToLineSegmentDistance(const Eigen::Vector2d& p, const Eigen::Vector2d& v, const Eigen::Vector2d& w);


    // Variable
    std::shared_ptr<MappingParams> params_;
    std::vector<Cone> cone_memory_; // Memory of global cones
    mutable std::mutex cone_memory_mutex_;
    std::vector<Eigen::Vector2d> left_lane_points_;
    std::vector<Eigen::Vector2d> right_lane_points_;
};

// Planning
class StateMachine {
public:
    StateMachine();
    ~StateMachine() = default;

    // State management
    ASState getCurrentState() const { return current_state_; }
    std::string getCurrentStateString() const { return stateToString(current_state_); }
    bool isValidTransition(ASState from, ASState to) const;
    
    // Event processing
    StateTransitionResult processEvent(ASEvent event);
    
    // Mission management
    std::string getCurrentMission() const { return current_mission_; }
    
    // Event injection (instead of ROS callbacks)
    void injectSystemInit();
    void injectGoSignal(const std::string& mission, const std::string& track);
    
    // Debug and monitoring
    void printStateInfo() const;
    double getTimeInCurrentState() const;
    
private:
    // State transition handlers
    bool enterAS_OFF();
    bool enterAS_READY();
    bool enterAS_DRIVING();
    bool enterAS_FINISHED();
    
    bool exitAS_OFF();
    bool exitAS_READY();
    bool exitAS_DRIVING();
    bool exitAS_FINISHED();
    
    // Internal state management
    bool performStateTransition(ASState new_state, const std::string& reason);
    
    // Utility functions
    std::string stateToString(ASState state) const;
    std::string eventToString(ASEvent event) const;
    void initializeValidTransitions();
    void logStateTransition(ASState from, ASState to, const std::string& reason);
    
private:
    // State management
    ASState current_state_;
    ASState previous_state_;
    std::chrono::steady_clock::time_point state_entry_time_;
    std::chrono::steady_clock::time_point last_update_time_;
    
    // Mission information
    std::string current_mission_;
    std::string mission_track_;
    bool mission_active_;
    
    // Valid state transitions (finite state machine definition)
    std::map<std::pair<ASState, ASState>, bool> valid_transitions_;
};

class TrajectoryGenerator {
public:
    explicit TrajectoryGenerator(const std::shared_ptr<PlanningParams>& params);
    ~TrajectoryGenerator() = default;
    
    // Generate trajectory from cones and current planning state(for Mapping mode)
    std::vector<TrajectoryPoint> generatePathFromClosestCones(const std::vector<Cone>& cones, const PlanningParams::TrajectoryModeParams& params);
    // Generate local trajectory by following the global path (for RACING mode)
    std::vector<TrajectoryPoint> getTrajectoryFromGlobalPath(const VehicleState& vehicle_state, const std::vector<TrajectoryPoint>& global_path, const PlanningParams::TrajectoryModeParams& params);
    std::vector<TrajectoryPoint> generateStopTrajectory();

    /**
     * @brief Get last generated trajectory
     * @return Last trajectory
     */
    const std::vector<TrajectoryPoint>& getLastTrajectory() const { return last_trajectory_; }
    const std::vector<Eigen::Vector2d>& getLastLocalPathPoints() const { return last_local_path_points_; }

    /**
     * @brief Update trajectory parameters
     * @param params New parameters
     */
    void updateParams(const std::shared_ptr<PlanningParams>& params) { params_ = params; }
    
    const std::shared_ptr<PlanningParams>& getParams() const { return params_; }
    
    /**
     * @brief Print trajectory statistics
     */
    void printTrajectoryStats() const;

private:
    // Utility functions
    double calculateCurvature(const tk::spline& s, double x);
    double calculateDistance(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2) const;
    double calculateAngle(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2) const;

    // Member variables
    std::shared_ptr<PlanningParams> params_;
    std::vector<TrajectoryPoint> last_trajectory_;
    std::vector<Eigen::Vector2d> last_local_path_points_;
    
    // Statistics
    int generated_trajectories_;
    double average_generation_time_;
    double last_generation_time_;
};

// Control
// Lateral Controller
class LateralController 
{
public:
    virtual ~LateralController() = default;
    virtual double calculateSteeringAngle(const VehicleState& current_state,
                                          const std::vector<TrajectoryPoint>& path,
                                          const ControlParams::ControllerModeParams& params,
                                          const double& vehicle_length) const = 0;
};

// Pure Pursuit Controller
class PurePursuit : public LateralController
{
public:
    explicit PurePursuit(); // No longer gets parameters
    
    double calculateSteeringAngle(const VehicleState& current_state,
                                  const std::vector<TrajectoryPoint>& path,
                                  const ControlParams::ControllerModeParams& params,
                                  const double& vehicle_length) const override;

private:
    // 내부 헬퍼 함수들은 그대로 유지
    int findTargetPointIndex(const std::vector<TrajectoryPoint>& path, const ControlParams::ControllerModeParams& params) const;
    double calculateSteeringAngleInternal(const Eigen::Vector2d& target_point,
                                        const ControlParams::ControllerModeParams& params,
                                        const double& vehicle_length) const;
};

// Stanley Controller
class Stanley : public LateralController
{
public:
    explicit Stanley(); // No longer gets parameters

    double calculateSteeringAngle(const VehicleState& current_state,
                                  const std::vector<TrajectoryPoint>& path,
                                  const ControlParams::ControllerModeParams& params,
                                  const double& vehicle_length) const override;

private:
    mutable double last_filtered_steering_angle_ = 0.0;
};

/**
 * @class PIDController
 * @brief 표준 PID(Proportional-Integral-Derivative) 제어기 클래스
 */
class PIDController
{
public:
    PIDController(); // No longer gets parameters

    /**
     * @brief 제어값을 계산합니다.
     * @param setpoint 목표값 (Desired value)
     * @param measured_value 현재 측정값 (Actual value)
     * @param kp 비례(Proportional) 게인
     * @param ki 적분(Integral) 게인
     * @param kd 미분(Derivative) 게인
     * @param max_output 최대 출력값
     * @return double 계산된 제어 출력값
     */
    double calculate(double setpoint, double measured_value, double kp, double ki, double kd, double max_output);

    /**
     * @brief 제어기의 내부 상태(적분항, 이전 오차)를 초기화합니다.
     */
    void reset();

private:
    // 제어기 내부 상태 변수
    double integral_error_;
    double previous_error_;
    
    // 시간 변화량(dt) 계산을 위한 변수
    std::chrono::steady_clock::time_point last_time_;
    bool first_run_;
};

// ====================================================================
// ==================== Formula Autonomous System =====================
// ====================================================================

class FormulaAutonomousSystem
{
public:
    FormulaAutonomousSystem();
    ~FormulaAutonomousSystem();

// Functions
public:
    bool init(ros::NodeHandle& pnh);
    bool getParameters();

    std::vector<Cone> getGlobalConeMap() const;
    int getCurrentLap() const { return current_lap_; }
    std::pair<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>> getGlobalTrackLanes() const {
        if (map_manager_) {
            return map_manager_->getTrackLanes();
        }
        return {};
    }
    
    Eigen::Vector2d getStartFinishLineCenter() const { return start_finish_line_center_; }
    Eigen::Vector2d getStartFinishLineDirection() const { return start_finish_line_direction_; }
    bool isStartFinishLineDefined() const { return is_start_finish_line_defined_; }

    // Globalpath
    std::vector<TrajectoryPoint> getGlobalPath() const { return global_path_; }
    bool isGlobalPathGenerated() const { return is_global_path_generated_; }
    const std::unique_ptr<TrajectoryGenerator>& getTrajectoryGenerator() const { return trajectory_generator_; }
    const std::vector<std::pair<size_t, size_t>>& getCriticalSections() const { return critical_sections_; }

    private: // Main thread

public: // Function components
    bool run(sensor_msgs::PointCloud2& lidar_msg,
             sensor_msgs::Image& camera1_msg,
             sensor_msgs::Image& camera2_msg,
             sensor_msgs::Imu& imu_msg,
             sensor_msgs::NavSatFix& gps_msg,
             fs_msgs::GoSignal& go_signal_msg,
             fs_msgs::ControlCommand& control_command_msg,
             std_msgs::String& autonomous_mode_msg);
            
private:
    void getLidarPointCloud(sensor_msgs::PointCloud2& msg, pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud);
    void getCameraImage(sensor_msgs::Image& msg, cv::Mat& image);
    void getImuData(sensor_msgs::Imu& msg, Eigen::Vector3d& acc, Eigen::Vector3d& gyro, Eigen::Quaterniond& orientation);
    void setRacingStrategy(const VehicleState& vehicle_state, const std::vector<Cone>& cones_for_planning);
    void defineStartFinishLine(const VehicleState& vehicle_state, const std::vector<Cone>& cones);
    void updateLapCount(const VehicleState& current_state);
    void updateVehiclePositionRelativeToLine(const VehicleState& current_state);
    void generateGlobalPath();

// Variables
private:
    bool is_initialized_;
    ros::NodeHandle pnh_;

    // Perception
    std::shared_ptr<PerceptionParams> perception_params_;
    std::unique_ptr<RoiExtractor> roi_extractor_;
    std::unique_ptr<GroundRemoval> ground_removal_;
    std::unique_ptr<Clustering> clustering_;
    
    std::unique_ptr<ColorDetection> color_detection_;

    // Localization
    std::shared_ptr<LocalizationParams> localization_params_;

    // Mapping
    std::shared_ptr<MappingParams> mapping_params_;
    std::unique_ptr<MapManager> map_manager_;

    // State machine
    std::unique_ptr<StateMachine> state_machine_;
    ASState planning_state_;

    // Driving mode and lap counting
    DrivingMode current_mode_;
    int current_lap_;
    bool is_race_finished_;

    Eigen::Vector2d start_finish_line_center_;
    Eigen::Vector2d start_finish_line_direction_;
    double start_finish_line_yaw_;
    bool is_start_finish_line_defined_;

    bool just_crossed_line_;
    double vehicle_position_relative_to_line_;
    double previous_position_relative_to_line_;
    ros::Time last_lap_time_;

    // Trajectory planning
    std::shared_ptr<PlanningParams> planning_params_;
    std::unique_ptr<TrajectoryGenerator> trajectory_generator_;
    std::vector<TrajectoryPoint> global_path_;
    std::vector<std::pair<size_t, size_t>> critical_sections_;
    bool is_global_path_generated_;

    // Control
    std::shared_ptr<ControlParams> control_params_;
    std::unique_ptr<LateralController> lateral_controller_;
    std::unique_ptr<PIDController> longitudinal_controller_;

public:
    // Odometry & TF broadcasting
    std::unique_ptr<Localization> localization_;

    // PointCloud for visualization
    pcl::PointCloud<pcl::PointXYZ>::Ptr roi_point_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_point_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground_point_cloud_;
    
    // Cone data for visualization
    std::vector<Cone> cones_;
    cv::Mat projected_cones_image_;
    
    // Trajectory for visualization
    std::vector<TrajectoryPoint> trajectory_points_;
};

#endif // FORMULA_AUTONOMOUS_SYSTEM_HPP