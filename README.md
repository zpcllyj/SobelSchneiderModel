# Sobel-Schneider Model

Single layer model based on Sobel and Schneider 2009 and 2013. 

Model equations:
$$\partial_t u - v ( \beta y - \partial_y u ) = \mathcal{H} (\partial_y v)  (\partial_y v) u - \mathcal{F} - \mathcal{S},$$
$$2\partial_t v +\beta y u = - \frac{gH}{T_0}\partial_y T,$$
$$\partial_t \theta+\frac{\delta \Delta_z}{H}\partial_y v=\frac{\theta_E-\theta }{\tau}.$$

# Reference

Sobel, A. H., and Schneider, T. (2009), Single-layer axisymmetric model for a Hadley circulation with parameterized eddy momentum forcing, J. Adv. Model. Earth Syst., 1, 10, doi:10.3894/JAMES.2009.1.10.

Sobel, A. H., and Schneider, T. (2013), Correction to “Single-layer axisymmetric model for a Hadley circulation with parameterized eddy momentum forcing”, J. Adv. Model. Earth Syst., 5, 654– 657, doi:10.1002/jame.20030.