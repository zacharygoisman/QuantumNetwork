# QuantumNetwork
Research project for optimizing quantum network routing and resource allocation, with the eventual hope of creating a physical system.

We have two goals at the moment:
1. Optimize channel allocation within a source to all links using said source.
2. Find the best source and path for each pair of users.

This may be expanded on later when considering a dynamic network.

**How does this system work?**  
Focusing on the first goal, say we have one source and all **L** user pairs (called links) wish to connect to that source. The source has some finite number of channels **K** it can allocate to these links. These channels all have some flux value **μ** determined by the source. Each link has a fidelity limit, say **0.7**, that is a lower bound on the fidelity each link can have, where the fidelity of a link is calculated from the following equation:

$$F = 0.25 \cdot \left( 1 + \frac{3x}{x^2 + x(2y_1 + 2y_2 + 1) + 4y_1 y_2} \right)$$

**y₁** and **y₂** are noise constants that depend on constants like efficiency that are related to each individual link. **x** is the total fidelity being allocated to each link based on the channels, and is a function of **μ**. This means that there is a limit to the amount of flux and by extension channels one can allocate to a link before it goes below the fidelity limit. Based on the equation, we can see that generally a larger flux causes a smaller fidelity. However, the main optimization goal is to maximize rate, which is calculated as such:

$$R = x^2 + x(2y_1 + 2y_2 + 1) + 4y_1 y_2$$

As we can see, the rate **R** generally is inversely proportional to the fidelity **F**, so we want to essentially maximize our allocated flux to the link or make the fidelity of the link as close to its limit as possible. This problem is a variation of the multiple knapsack problem.

The second optimizer focuses on determining which source is best for a link to use based on the loss values it experiences. This ends up being a simpler problem, and we can use methods like a Dijkstra algorithm between the source and each user to determine the path with the lowest loss.
