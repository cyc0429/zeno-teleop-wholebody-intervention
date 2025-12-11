from pymlg import SE3 
import numpy as np

# Random pose
T = SE3.random()

# R^n to group directly (using "Capital" notation)
x = np.array([0.1, 0.2, 0.3, 4, 5, 6])
T = SE3.Exp(x)
print("T: ", T)
# Group to R^n directly
x = SE3.Log(T)
print("x: ", x)
# Wedge, vee
Xi = SE3.wedge(x)
x = SE3.vee(Xi)

# Actual exp/log maps 
T = SE3.exp(Xi)
Xi = SE3.log(T)

# Adjoint matrix representation of group element
A = SE3.adjoint(T)

# Adjoint representation of algebra element
ad = SE3.adjoint_algebra(Xi)

# Inverse of group element
T_inv = SE3.inverse(T)

# Group left/right jacobians, and their inverses
J_L = SE3.left_jacobian(x)
J_R = SE3.right_jacobian(x)
J_L_inv = SE3.left_jacobian_inv(x)
J_R_inv = SE3.right_jacobian_inv(x)

# ... and more.
