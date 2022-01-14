# External resources {#ext-resources}

## Anderson acceleration

- [Donald G.M. Anderson, “Iterative procedures for nonlinear integral equations”. _Journal of the Association for Computing Machinery_, Vol. 12, No. 4, pp. 547─560, 1965.](https://dl.acm.org/doi/pdf/10.1145/321296.321305)  
  Original paper introducing the idea of Anderson acceleration, not that 
  relevant for the version we implemented.
- [Donald G.M. Anderson, Comments on “Anderson Acceleration, Mixing and Extrapolation,” _Numerical Algorithms_, Vol. 80, No. 1: pp. 135─234, 2017.](http://nrs.harvard.edu/urn-3:HUL.InstRepos:34773632)  
  Long but interesting discussion of Anderson acceleration by Anderson himself,
  also discussing other publications on the topic.
- [Homer F. Walker and Peng Ni, “Anderson acceleration for fixed-point iterations,” _SIAM Journal on Numerical Analysis_, Vol. 49, No. 4, pp. 1715─1735, 2011.](https://users.wpi.edu/~walker/Papers/Walker-Ni,SINUM,V49,1715-1735.pdf)  
  Anderson acceleration applied to fixed-point iteration, covers the math used
  for the implementation of our algorithm.  
  PDF with pseudocode and Mᴀᴛʟᴀʙ implementation:
  https://users.wpi.edu/~walker/Papers/anderson_accn_algs_imps.pdf
<!-- - [Junzi Zhang, Brendan O’Donoghue, and Stephen Boyd, “Globally convergent type-I Anderson acceleration for nonsmooth fixed-point iterations,” _SIAM Journal on Optimization_, Vol. 30, No. 4, pp. 3170─3197, 2020.](https://stanford.edu/~boyd/papers/pdf/scs_2.0_v_global.pdf)  
  Looks interesting, quite sophisticated algorithm, haven't really looked at it
  in great detail. (IIRC, Boyd has more papers on the topic.) -->
- [Vien V. Mai and Mikael Johansson, “Anderson Acceleration of Proximal Gradient Methods,” _arXiv:1910.08590_, 2020.](https://arxiv.org/abs/1910.08590)  
  Guarded Anderson acceleration applied to the proximal gradient algorithm.
- [J.W. Daniel, W.B. Gragg, L. Kaufman and G.W. Stewart, “Reorthogonalization and stable algorithms for updating the Gram-Schmidt QR factorization,” _Mathematics of computation_, Vol. 30, No. 136, pp. 772─795, 1976.](https://www.ams.org/journals/mcom/1976-30-136/S0025-5718-1976-0431641-8/S0025-5718-1976-0431641-8.pdf)  
  Algorithm for incrementally updating the QR factorization as columns are added
  or removed.
<!-- - [Nicholas J. Higham, “A Survey of Condition Number Estimation for Triangular Matrices,” _SIAM Review_, Vol. 29, No. 4, pp. 575─596, 1987.](http://eprints.ma.man.ac.uk/695/1/covered/MIMS_ep2007_10.pdf)  
  Not sure if this is very useful. -->
- [Christian Bischof, Ping Tak Peter Tang, “Robust Incremental Condition Estimation,” _SIAM Journal on Matrix Analysis and Applications_, Vol. 11, No. 2, pp. 312─322, 1990.](https://epubs.siam.org/doi/abs/10.1137/0611021)  
  This is on the todo list: estimate the condition of the QR factorization and
  drop columns if it is ill-conditioned (close to singular).  
  Another, newer (unpublished?) version:
  https://www.osti.gov/biblio/10133022-Zz8cvq/native  
  Better quality PDF: http://www.netlib.org/lapack/lawnspdf/lawn33.pdf
