# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:16:33 2024

@author: user
"""

# =============================================================================
# Pareto 3D
# =============================================================================
# Function to identify Pareto optimal solutions
def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a default array of 'True' values
    pareto_front = np.ones(population_size, dtype=bool)
    for i in range(population_size):
        for j in range(population_size):
            # Check if our 'i' point is dominated by our 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i
                pareto_front[i] = 0
                break
    return pareto_front

scores = np.stack((x, y, z), axis=1)
pareto = identify_pareto(scores)
pareto_front = scores[pareto] # pareto : index

fig = plt.figure(figsize =(10,10), dpi = 300)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, marker = 'o', color = 'w', edgecolor = 'k', alpha = 0.5)
ax.scatter(x_, y_, z_, marker = '^', color = 'w', edgecolor = 'k', alpha = 0.5)
ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2],s = 30, color='#0047BB', marker = 's')

# Delaunay triangulation
tri = Delaunay(pareto_front[:, :2])

# Plot the surface
ax.plot_trisurf(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], triangles=tri.simplices, color='#0047BB', alpha=0.2)
ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2],s = 30, color='#0047BB', marker = 's')

# ax.set_facecolor('white')
ax.view_init(elev=12, azim=80+170)
ax.set_xlabel('Stiffness', labelpad=20) 
ax.set_ylabel('Strength', labelpad=20) 
ax.set_zlabel('Toughness', labelpad=20) 
# plt.title('3D Pareto optimal solutions in a randomly generated dataset')
# ax.plot(pareto_front[:, 0], pareto_front[:, 1], 'r+', zdir='y', zs=1.5)

ax.set_xlim([150,500])
ax.set_zlim([1e-05,3.3e-05])
plt.show()

#%%
# =============================================================================
# Pareto 2D
# =============================================================================
import matplotlib.patheffects as mpe
outline=mpe.withStroke(linewidth=5.5, foreground='w')

plt.figure(figsize=(6.3, 5), dpi = 300)
plt.scatter(x, y, color='w',s = 20,edgecolors='k', marker = '^',alpha = 0.5,label='Data Points')
plt.scatter(x_, y_, color='w',s = 20,marker = 'o',edgecolors='k',alpha = 0.5, label='initial design space')
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c = pareto_front[:,2],cmap = 'RdYlBu_r', s = 30,marker = 's', label='Pareto Optimal', path_effects = [outline])
clb = plt.colorbar()
clb.set_label('toughness', labelpad=-40, x = 1,y=1.125, rotation=0)
# Sort the Pareto optimal points by x-coordinate
pareto_x, pareto_y = zip(*sorted(zip(pareto_front[:, 0], pareto_front[:, 1])))

plt.xlabel('Stiffness')
plt.ylabel('strength')

plt.show()

plt.figure(figsize=(6.3, 5), dpi = 300)
plt.scatter(x, z, color='w',s = 20,edgecolors='k', alpha = 0.5,marker = '^',label='Data Points')
plt.scatter(x_, z_, color='w',s = 20,marker = 'o',edgecolors='k',alpha = 0.5, label='initial design space')
plt.scatter(pareto_front[:, 0], pareto_front[:, 2], c = pareto_front[:,1],cmap = 'RdYlBu_r', s = 30,marker = 's', label='Pareto Optimal', path_effects = [outline])
clb = plt.colorbar()
clb.set_label('Strength', labelpad=-40, y=1.075, rotation=0)

# Sort the Pareto optimal points by x-coordinate
pareto_x, pareto_y = zip(*sorted(zip(pareto_front[:, 0], pareto_front[:, 2])))

plt.xlabel('Stiffness')
plt.ylabel('Toughness')
plt.show()


plt.figure(figsize=(6.3, 5), dpi = 300)
plt.scatter(y, z, color='w',s = 20,edgecolors='k',alpha = 0.5, marker = '^',label='Data Points')
plt.scatter(y_, z_, color='w',s = 20,marker = 'o',alpha = 0.5,edgecolors='k', label='initial design space')
plt.scatter(pareto_front[:, 1], pareto_front[:, 2], c = pareto_front[:,0],cmap = 'RdYlBu_r', s = 30,marker = 's', label='Pareto Optimal', path_effects = [outline])
clb = plt.colorbar()
clb.set_label('Stiffness', labelpad=-40, y=1.075, rotation=0)

# Sort the Pareto optimal points by x-coordinate
pareto_x, pareto_y = zip(*sorted(zip(pareto_front[:, 1], pareto_front[:, 2])))

plt.xlabel('Strength')
plt.ylabel('Toughness')
plt.show()

