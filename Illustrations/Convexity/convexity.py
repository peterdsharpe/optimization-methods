import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 300)


def plot(subplot, expression, color):
    plt.subplot(subplot)
    plt.plot(x, expression, color)
    plt.grid(True)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$")

plot(221,
     0.5 * (x - 2) ** 2 - 3,
     'b'
     )
plot(222,
     x**2/5+2/np.sqrt(1+(x-2)**2)-2,
     'r'
     )
plot(223,
     np.abs(x)-2,
     'b'
     )
plot(224,
     x**2/5+2/np.sqrt(1+(x-2)**2)-2-0.5*x,
     'r'
     )
plt.tight_layout()
plt.savefig("convexity.png", dpi=600)
plt.show()

### Aero
import aerodynamics as aero

Re = np.logspace(4, 8, 300)
Cf = aero.Cf_flat_plate(Re)
Cf_conv = aero.Cf_flat_plate_convex(Re)

plt.loglog(Re, Cf, label="Blasius Soln. + Prandtl")
plt.loglog(Re, Cf_conv, label="Schlichting Approx.")
plt.grid(True)
plt.xlabel("Reynolds Number")
plt.ylabel("Skin Friction Coefficient")
plt.title("Flat Plate Skin Friction Coefficient")
plt.legend()
plt.tight_layout()
plt.savefig("flatplate.png", dpi=600)
plt.show()
