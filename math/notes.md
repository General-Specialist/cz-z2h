# Multivariable Calculus Notes & Formulas

Welcome to the comprehensive Multivariable Calculus study sheet, transcribed from your handwritten formulas and text notes.

---

## 1. Vectors, Products, and Projections

### Dot Product vs. Cross Product

> **Note: Conceptual Comparison**
> 
> * **Dot Product** ($\vec{a} \cdot \vec{b}$):
>   * Returns a **scalar** (a single real number).
>   * Measures the extent to which two vectors point in the same direction.
>   * **Algebraically**: $\vec{a} \cdot \vec{b} = a_x b_x + a_y b_y + a_z b_z$
>   * **Geometrically**: $\vec{a} \cdot \vec{b} = |\vec{a}| |\vec{b}| \cos(\theta)$, where $\theta$ is the angle between the vectors.
>   * Two vectors are **perpendicular** (orthogonal) if and only if their dot product is zero:
>     $$\vec{a} \cdot \vec{b} = 0 \iff \vec{a} \perp \vec{b}$$
> 
> * **Cross Product** ($\vec{a} \times \vec{b}$):
>   * Returns a **vector** in $\mathbb{R}^3$ that is perpendicular to both $\vec{a}$ and $\vec{b}$ (direction determined by the Right-Hand Rule).
>   * Measures the extent to which two vectors are perpendicular.
>   * **Geometrically**: $|\vec{a} \times \vec{b}| = |\vec{a}| |\vec{b}| \sin(\theta)$, where $\theta$ is the angle between the vectors.
>   * Two vectors are **parallel** if and only if their cross product is the zero vector:
>     $$\vec{a} \times \vec{b} = \vec{0} \iff \vec{a} \parallel \vec{b}$$

#### Geometric Applications of the Cross Product

* **Area of a Parallelogram** spanned by vectors $\vec{a}$ and $\vec{b}$:
  $$\text{Area} = |\vec{a} \times \vec{b}|$$
* **Area of a Triangle** formed by adjacent side vectors $\vec{a}$ and $\vec{b}$:
  $$\text{Area} = \frac{1}{2} |\vec{a} \times \vec{b}|$$

---

### Vector Projections

Vector projection of a vector $\vec{b}$ onto a vector $\vec{a}$ (written as "$\vec{b}$ onto $\vec{a}$") finds the component vector of $\vec{b}$ pointing along the line of $\vec{a}$.

$$\text{proj}_{\vec{a}} \vec{b} = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}|} \cdot \frac{\vec{a}}{|\vec{a}|} = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}|^2} \vec{a}$$

This projection is composed of:
1. **Scalar Projection** (the component magnitude of $\vec{b}$ along $\vec{a}$):
   $$\text{comp}_{\vec{a}} \vec{b} = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}|}$$
2. **Direction Vector** (the unit vector of $\vec{a}$):
   $$\vec{u}_{\vec{a}} = \frac{\vec{a}}{|\vec{a}|}$$

---

## 2. Lines and Planes in 3D Space

### Equation of a Plane & The Normal Vector

A plane in 3D space is uniquely determined by a point $P_0(x_0, y_0, z_0)$ on the plane and a vector $\vec{n} = \langle a, b, c \rangle$ perpendicular (orthogonal) to the plane. This vector $\vec{n}$ is the **normal vector**.

For any arbitrary point $P(x, y, z)$ on the plane, the vector $\vec{P_0 P} = \langle x - x_0, y - y_0, z - z_0 \rangle$ lies entirely within the plane. Since $\vec{n}$ is orthogonal to the plane, it must be orthogonal to $\vec{P_0 P}$, meaning their dot product is zero:

$$\vec{n} \cdot \vec{P_0 P} = 0$$

$$\langle a, b, c \rangle \cdot \langle x - x_0, y - y_0, z - z_0 \rangle = 0$$

$$a(x - x_0) + b(y - y_0) + c(z - z_0) = 0$$

---

### Point of Intersection of Two Planes

If two planes are not parallel, they must intersect along a straight line.

> **Tip: Method to Find the Line of Intersection**
> 1. **Find the direction vector ($\vec{v}$)** of the line by computing the cross product of the normal vectors of the two planes ($\vec{n}_1$ and $\vec{n}_2$):
>    $$\vec{v} = \vec{n}_1 \times \vec{n}_2$$
> 2. **Set one variable to zero** (usually $z = 0$ or $x = 0$) to reduce the system of two plane equations to two equations with two variables.
> 3. **Solve the system of equations** to find the remaining coordinates, yielding a point $P_0(x_0, y_0, z_0)$ on the line.
> 4. **Combine the point and direction vector** to write the equation of the line:
>    $$\vec{r}(t) = \vec{r}_0 + t\vec{v}$$

#### Worked Example
* **Problem**: A line $l$ passes through the point $(-1, 1, 2)$ and is perpendicular to the plane $x - 2y + 2z = 8$. At what point does this line intersect with the $yz$-plane?

* **Solution**:
  1. **Find the direction vector of the line ($\vec{v}$)**:
     The plane equation is $x - 2y + 2z = 8$. The coefficients yield the normal vector:
     $$\vec{n} = \langle 1, -2, 2 \rangle$$
     Since the line is perpendicular to the plane, the line's direction vector $\vec{v}$ is parallel to the normal vector $\vec{n}$. We can choose:
     $$\vec{v} = \vec{n} = \langle 1, -2, 2 \rangle$$
  2. **Write the equation of the line ($\vec{r}(t)$)**:
     Using the point $P_0(-1, 1, 2)$ and direction vector $\vec{v}$:
     $$\vec{r}(t) = \langle -1, 1, 2 \rangle + t\langle 1, -2, 2 \rangle$$
     Which gives the parametric equations:
     $$x(t) = -1 + t, \quad y(t) = 1 - 2t, \quad z(t) = 2 + 2t$$
  3. **Determine intersection with the $yz$-plane**:
     The line intersects the $yz$-plane where $x = 0$:
     $$x(t) = -1 + t = 0 \iff t = 1$$
  4. **Solve for the coordinates**:
     Substitute $t = 1$ back into the equations for $y(t)$ and $z(t)$:
     $$y(1) = 1 - 2(1) = -1$$
     $$z(1) = 2 + 2(1) = 4$$
     Thus, the intersection point is **$(0, -1, 4)$**.

---

## 3. Vector-Valued Functions and Space Curves

### Basic Definitions

* **Vector-Valued Function**: A function with a real number input (usually parameter $t$) and a vector output:
  $$\vec{r}(t) = \langle x(t), y(t), z(t) \rangle$$
* **Function of Several Variables**: A function with a point $(x, y)$ or $(x, y, z)$ as input and a single scalar number as output:
  $$z = f(x, y) \quad \text{or} \quad w = f(x, y, z)$$

---

### The TNB Frame (Tangent, Normal, and Binormal Vectors)

For a smooth space curve traced by $\vec{r}(t)$, we define three mutually orthogonal unit vectors:

* **Unit Tangent Vector** ($\vec{T}$) --- Points in the direction of motion:
  $$\vec{T}(t) = \frac{\vec{v}(t)}{|\vec{v}(t)|} = \frac{\vec{r}'(t)}{|\vec{r}'(t)|}$$
* **Unit Normal Vector** ($\vec{N}$) --- Points in the direction the curve is bending (orthogonal to $\vec{T}$):
  $$\vec{N}(t) = \frac{\vec{T}'(t)}{|\vec{T}'(t)|}$$
* **Binormal Vector** ($\vec{B}$) --- Completes the right-handed frame (orthogonal to both $\vec{T}$ and $\vec{N}$):
  $$\vec{B}(t) = \vec{T}(t) \times \vec{N}(t)$$

---

### Reparameterizing by Arc Length

Reparameterization by arc length expresses the curve $\vec{r}$ in terms of the distance traveled ($s$) along the curve rather than time ($t$), resulting in a constant unit speed traversal ($|\vec{r}'(s)| = 1$).

1. **Start with the function**: $\vec{r}(t) = \langle x(t), y(t), z(t) \rangle$ for $t \in [a, b]$.
2. **Calculate speed** $|\vec{r}'(t)|$:
   $$|\vec{r}'(t)| = \sqrt{(x'(t))^2 + (y'(t))^2 + (z'(t))^2}$$
3. **Define the arc-length function** $s(t)$ by integrating speed:
   $$s(t) = \int_{a}^{t} |\vec{r}'(u)| \, du$$
   This represents the distance traveled starting from $t = a$.
4. **Solve for $t$** in terms of $s$:
   $$t = t(s)$$
5. **Substitute** $t(s)$ back into the original vector function:
   $$\vec{r}(s) = \vec{r}(t(s))$$

---

### Curvature

Curvature ($\kappa$, kappa) measures how sharply a curve bends or changes direction at a given point.

$$\kappa(t) = \frac{|\vec{T}'(t)|}{|\vec{r}'(t)|}$$

$$\kappa(t) = \frac{|\vec{v}(t) \times \vec{a}(t)|}{|\vec{v}(t)|^3} = \frac{|\vec{r}'(t) \times \vec{r}''(t)|}{|\vec{r}'(t)|^3}$$

> **Note:** Curvature is a geometric property of the curve and is independent of the parameterization.
> Curvature is **always non-negative** ($\kappa(t) \ge 0$).

---

## 4. Quadric Surfaces

Quadric surfaces are the 3D graphs of second-degree equations in $x, y,$ and $z$. 

| Function $z = f(x,y)$ | Equation | Surface Type |
| :--- | :--- | :--- |
| $f(x,y) = x^2 + y^2$ | $z = x^2 + y^2$ | Paraboloid (Circular) |
| $f(x,y) = x^2 - y^2$ | $z = x^2 - y^2$ | Saddle (Hyperbolic Paraboloid) |
| $f(x,y) = \sqrt{x^2 + y^2}$ | $z = \sqrt{x^2 + y^2}$ | Cone (Circular) |

> **Important: Additional Quadric Surfaces (from reference materials, Page 6)**
> * **Ellipsoid**: $\frac{x^2}{a^2} + \frac{y^2}{b^2} + \frac{z^2}{c^2} = 1$
> * **Hyperboloid of One Sheet**: $\frac{x^2}{a^2} + \frac{y^2}{b^2} - \frac{z^2}{c^2} = 1$
> * **Hyperboloid of Two Sheets**: $-\frac{x^2}{a^2} - \frac{y^2}{b^2} + \frac{z^2}{c^2} = 1$
> * **Elliptic Cone**: $\frac{z^2}{c^2} = \frac{x^2}{a^2} + \frac{y^2}{b^2}$
> * **Elliptic Paraboloid**: $\frac{z}{c} = \frac{x^2}{a^2} + \frac{y^2}{b^2}$
> * **Hyperbolic Paraboloid**: $\frac{z}{c} = \frac{x^2}{a^2} - \frac{y^2}{b^2}$
