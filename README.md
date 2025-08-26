# QuantumNetwork
Research project for optimizing quantum network routing and resource allocation, with the eventual hope of creating a physical system.

We have two goals at the moment:
1. Optimize channel allocation within a source to all links using said source. (Resource Allocation)
2. Find the best source and path for each pair of users. (Routing)

This may be expanded on later when considering a dynamic network or a physical system.

**Definitions**
We define that a source is a node that sends the entangled bits to the users. A link is established between two users if the source sends entangled bits to each user pair. Channels are the frequencies that a source allocates to the link pairs, where each frequency channel has some flux, representing the entangled photon rate. Lastly, a path is the route in a network that a link uses.

**How does this system work?**  
Focusing on the first goal, say we have one source and all **L** user pairs (called links) wish to connect to that source. The source has some finite number of channels **K** it can allocate to these links. These channels all have some flux value **μ** determined by the source. Each link has a fidelity limit, say **0.7**, that is a lower bound on the fidelity each link can have, where the fidelity of a link is calculated from the following equation:

$$F = 0.25 \cdot \left( 1 + \frac{3x}{x^2 + x(2y_1 + 2y_2 + 1) + 4y_1 y_2} \right)$$

**y₁** and **y₂** are noise constants that depend on constants like efficiency that are related to each individual link. **x** is the total fidelity being allocated to each link based on the channels, and is a function of **μ**. This means that there is a limit to the amount of flux and by extension channels one can allocate to a link before it goes below the fidelity limit. Based on the equation, we can see that generally a larger flux causes a smaller fidelity. However, the main optimization goal is to maximize rate, which is calculated as such:

$$R = x^2 + x(2y_1 + 2y_2 + 1) + 4y_1 y_2$$

As we can see, the rate **R** generally is inversely proportional to the fidelity **F**, so we want to essentially maximize our allocated flux to the link or make the fidelity of the link as close to its limit as possible. This problem is a variation of the multiple knapsack problem. In order to solve it, we use the APOPT optimizer, a nonlinear programming solver, to determine the best allocation of channels to links by solving the following:

$$
\begin{aligned}
&\max_{\mu, k_i} \quad \sum \log_{10}\big(\mu^2 k_i^2 + \mu k_i(2y_{1_i} + 2y_{2_i} + 1) + 4y_{1_i}y_{2_i}\big) \\
&\text{s.t.} \\
&\quad 0.25\Bigg(1 + \Bigg(\frac{3\mu k_i}{\mu^2 k_i^2 + \mu k_i(2y_{1_i} + 2y_{2_i} + 1) + 4y_{1_i}y_{2_i}}\Bigg)\Bigg) \ge f_i, \\
&\quad \sum k_i \le K, \\
&\quad \mu, k_i \ge 0, \\
&\quad k_i \in \mathbb{Z}.
\end{aligned}
$$

The second optimizer focuses on determining which source is best for a link to use based on the loss values it experiences. We use Dijkstra's algorithm between the source and each user to determine each link's lowest loss path to each of the sources. We make the assumption that all link paths must use the Dijkstra generated paths to a specific source to greatly simplify the problem for more complicated networks.

Lastly, we use the CP-SAT solver to check if our network has interfering frequency channels.
