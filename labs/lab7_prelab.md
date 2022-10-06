# Prelab 7
## linear system we need to solve

1 x_0 x_0^2 ... x_0^n-1 x_0^n <br>
1 x_1 x_1^2 ... x_1^n-1 x_1^n <br>
1 x_2 x_2^2 ... x_2^n-1 x_2^n <br>
... <br>
... <br>
... <br>
1 x_n x_n^2 ... x_n^n-1 x_n^n <br>

### Solving for the matrix
solving should not be hard, we can simply compute and inverse and multiply<br>
Ax=b, then x=bA^-1
### Evaluating the polynomial
if we are able to solve the system above, then we will have a vector of the coefficients<br>
evaluating the polynomial from here should not prove too difficult with some simple multiplication of
the solved for coefficients multiplied by the points. I don't see why a couple loops and the linalg numpy 
package shouldn't be able to handle most of this!
