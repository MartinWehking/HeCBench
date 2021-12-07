/*
  Purpose:

    MAIN is the main program for FEYNMAN_KAC_2D.

  Discussion:

    This program is derived from section 2.5, exercise 2.2 of Petersen and Arbenz.

    The problem is to determine the solution U(X,Y) of the following 
    partial differential equation:

      (1/2) Laplacian U - V(X,Y) * U = 0,

    inside the elliptic domain D:
 
      D = { (X,Y) | (X/A)^2+(Y/B)^2 <= 1 }
   
    with the boundary condition U(boundary(D)) = 1.

    The V(X,Y) is the potential function:

      V = 2 * ( (X/A^2)^2 + (Y/B^2)^2 ) + 1/A^2 + 1/B^2.

    The analytic solution of this problem is already known:

      U(X,Y) = exp ( (X/A)^2 + (Y/B)^2 - 1 ).

    Our method is via the Feynman-Kac Formula.

    The idea is to start from any (x,y) in D, and
    compute (x+Wx(t),y+Wy(t)) where 2D Brownian motion
    (Wx,Wy) is updated each step by sqrt(h)*(z1,z2),
    each z1,z2 are independent approximately Gaussian 
    random variables with zero mean and variance 1. 

    Each (x1(t),x2(t)) is advanced until (x1,x2) exits 
    the domain D.  

    Upon its first exit from D, the sample path (x1,x2) is stopped and a 
    new sample path at (x,y) is started until N such paths are completed.
 
    The Feynman-Kac formula gives the solution here as

      U(X,Y) = (1/N) sum(1 <= I <= N) Y(tau_i),

    where

      Y(tau) = exp( -int(s=0..tau) v(x1(s),x2(s)) ds),

    and tau = first exit time for path (x1,x2). 

    The integration procedure is a second order weak accurate method:

      X(t+h)  = [ x1(t) + sqrt ( h ) * z1 ]
                [ x2(t) + sqrt ( h ) * z2 ]

    Here Z1, Z2 are approximately normal univariate Gaussians. 

    An Euler predictor approximates Y at the end of the step

      Y_e     = (1 - h*v(X(t)) * Y(t), 

    A trapezoidal rule completes the step:

      Y(t+h)  = Y(t) - (h/2)*[v(X(t+h))*Y_e + v(X(t))*Y(t)].

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    31 May 2012

  Author:

    Original C 3D version by Wesley Petersen.
    C 2D version by John Burkardt.

  Reference:

    Peter Arbenz, Wesley Petersen,
    Introduction to Parallel Computing:
    A Practical Guide with Examples in C,
    Oxford, 2004,
    ISBN: 0-19-851577-4,
    LC: QA76.59.P47.
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "common.h"
#include "util.h"
#include "kernel.h"

int main ( int argc, char **argv )
{
  if (argc != 2) {
    printf("Usage: %s <iterations>\n", argv[0]); 
    return 1;
  }

  const int repeat = atoi(argv[1]);
  double a = 2.0;
  double b = 1.0;
  int dim = 2;
  double err;
  double h = 0.001;
  int N = 1000;
  int n_inside;
  int ni;
  int nj;
  double rth;
  int seed = 123456789;

  printf ( "\n" );

  printf ( "\n" );
  printf ( "FEYNMAN_KAC_2D:\n" );
  printf ( "\n" );
  printf ( "  Program parameters:\n" );
  printf ( "\n" );
  printf ( "  The calculation takes place inside a 2D ellipse.\n" );
  printf ( "  A rectangular grid of points will be defined.\n" );
  printf ( "  The solution will be estimated for those grid points\n" );
  printf ( "  that lie inside the ellipse.\n" );
  printf ( "\n" );
  printf ( "  Each solution will be estimated by computing %d trajectories\n", N );
  printf ( "  from the point to the boundary.\n" );
  printf ( "\n" );
  printf ( "    (X/A)^2 + (Y/B)^2 = 1\n" );
  printf ( "\n" );
  printf ( "  The ellipse parameters A, B are set to:\n" );
  printf ( "\n" );
  printf ( "    A = %f\n", a );
  printf ( "    B = %f\n", b );
  printf ( "  Stepsize H = %6.4f\n", h );

  // scaled stepsize.
  rth = sqrt ( ( double ) dim * h );

  // a > b
  nj = 128;
  ni = 1 + i4_ceiling ( a / b ) * ( nj - 1 );

  printf ( "\n" );
  printf ( "  X coordinate marked by %d points\n", ni );
  printf ( "  Y coordinate marked by %d points\n", nj );

  err = 0.0;
  n_inside = 0;

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  range<2> gws ((nj+15)/16*16, (ni+15)/16);
  range<2> lws (16, 16);

  buffer<double, 1> d_err (1);
  buffer<int, 1> d_n_inside (1);

  timestamp ( );
  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      auto acc = d_err.get_access<sycl_discard_write>(cgh);
      cgh.copy(&err, acc);
    });

    q.submit([&] (handler &cgh) {
      auto acc = d_n_inside.get_access<sycl_discard_write>(cgh);
      cgh.copy(&n_inside, acc);
    });

    q.submit([&] (handler &cgh) {
      auto err = d_err.get_access<sycl_read_write>(cgh);
      auto n = d_n_inside.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class solution>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
        fk (item, ni, nj, seed, N, a, b, h, rth, n.get_pointer(), err.get_pointer());
      });
    });
  }
  q.wait(); 
  timestamp ( );
  
  q.submit([&] (handler &cgh) {
    auto acc = d_err.get_access<sycl_read>(cgh);
    cgh.copy(acc, &err);
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_n_inside.get_access<sycl_read>(cgh);
    cgh.copy(acc, &n_inside);
  });

  q.wait(); 

  printf ( "  %e %d\n", err, n_inside );
  err = sqrt ( err / ( double ) ( n_inside ) );
  printf ( "\n" );
  printf ( "  RMS absolute error in solution = %e\n", err );
  printf ( "\n" );
  printf ( "FEYNMAN_KAC_2D:\n" );
  printf ( "  Normal end of execution.\n" );
  printf ( "\n" );

  return 0;
}
