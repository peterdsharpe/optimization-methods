import matplotlib.pyplot as plt
import matplotlib.style as style

style.use("seaborn")
plt.figure()
import numpy as np

x_vals = np.linspace(-5, 5, 500)

import sympy as sp

x = sp.Symbol("x")
f = x ** 2 / 5 + 2 / sp.sqrt(1 + (x - 2) ** 2) - 2
df = sp.diff(f, x)
ddf = sp.diff(f, x, 2)

f_func = sp.lambdify(x, f, "numpy")
df_func = sp.lambdify(x, df, "numpy")
ddf_func = sp.lambdify(x, ddf, "numpy")
f_vals = f_func(x_vals)

plt.figure(figsize=(2.5,2.5))
plt.plot(x_vals, f_vals, label="Objective", zorder=2)

global fig_count
fig_count = 0
fig_names = []
gamma = 1

def output():
    plt.xlim(-5.5, 5.5)
    plt.ylim(-1.5, 4)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$")
    plt.legend()
    # plt.tight_layout()
    # plt.title("1st-Order Gradient-Based Optimization Illustration")
    global fig_count
    fig_names.append("%i.png" % fig_count)
    plt.savefig(fig_names[-1], dpi = 200)
    fig_count += 1


max_steps = 30
xs = [4.]
fs = [f_func(xs[-1])]

for step in range(max_steps+1):
    # Draw with point
    plt.title(r"Gradient Descent: Iter %i" % (step))
    plt.plot(xs[-1], fs[-1], ".", color=(92/255, 0/255, 0/255), zorder = 4)
    output()

    # Draw tangent line
    dfdx = df_func(xs[-1])


    # Eval next point
    xs.append(
        xs[-1] - gamma * dfdx
    )
    fs.append(
        f_func(xs[-1])
    )

    # Break
    if np.abs(xs[-1] - xs[-2]) < 0.01:
        break

    plt.arrow(xs[-2], fs[-2], xs[-1]-xs[-2], fs[-1]-fs[-2], color=(227/255, 84/255, 84/255), width=0.03, length_includes_head=True, zorder = 3)

print("Finished after iteration %i" % step)

# Make a GIF
from PIL import Image
import glob

# Create the frames
frames = []
imgs = fig_names #glob.glob("*.png")
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save('gradient_descent_gamma_%.2f_%i_iters_small.gif' % (gamma, step), format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=300, loop=0)

# Delete all images
import os
[os.remove(file) for file in fig_names]