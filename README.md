# QuantumNetwork
Research project for optimizing quantum network routing and spectrum allocation, with the eventual of being used for a physical system.

We have two goals at the moment:
1. Optimize channel allocation within a source to all lightpaths using said source. (Spectrum Allocation)
2. Find the best source and route for each pair of users. (Routing)

This may be expanded on later when considering a dynamic network or a physical system.

**Definitions**
For simplicity, we restrict our topologies to have two types of nodes: sources and users. A **source** is a node that generates and sends entangled photon pairs to **user** nodes. These photon pairs are sent in the form of frequency bin **channels** which all have identical **flux** from the same source, and due to energy conservation, frequency channel 1 has a complementary channel -1 sent to the second user in the pair. Rather than have additional switch nodes, each node is also capable of forwarding these channels to other nodes. Furthermore, I will be describing an entangled link between two user pairs as a **link**.

**How does this system work?**  
After generating the topology, the pipeline has three phases to determine optimal routing and allocation. These are routing (using Double Yen), spectrum allocation (using APOPT), and frequency scheduling (using CP-SAT).

**Routing**
After collecting network information for the desired links, the pipeline generates the N lowest loss paths for each source to user, using Yen's algorithm for both sides of the link (hence the "Double Yen"). With these, it can find the lowest loss paths for each pair of users to reach every source. Ultimately, this will determine the combination of paths for all links that has the lowest loss. This order is the default order that the pipeline will run potential routing solutions for the next two phases.

**Spectrum Allocation**
To determine the best allocation of channels for each source used from the previous phase's combination, we need to determine a solution to the Entangled Flux Allocation problem. We use these three assumptions to focus on the RSA relevant features, as additional constraints can be added afterwards. These assumptions are:
1. Users are entangled with only one other
2. Identical quantum states are produced in all frequency bins
3. Channel distortion effects are fully compensated

These end up simplifying the state fidelity and total coincidence rate equations to be as follows:

**State Fidelity:**

$$
\mathcal{F}_\ell = \frac{1}{4}\left[ 1 + \frac{3\mu_{m_\ell} \tau K_\ell}{\mu_{m_\ell} \tau K_\ell + \left(\mu_{m_\ell} \tau K_\ell + \frac{2\tau d_{A_\ell}}{\eta_{A_\ell}}\right)\left(\mu_{m_\ell} \tau K_\ell + \frac{2\tau d_{B_\ell}}{\eta_{B_\ell}}\right)} \right]
$$

**Total Coincidence Rate:**

$$
\mathcal{R}_\ell = \mu_{m_\ell}\tau K_\ell + \left(\mu_{m_\ell} \tau K_\ell + \frac{2\tau d_{A_\ell}}{\eta_{A_\ell}}\right)\left(\mu_{m_\ell} \tau K_\ell + \frac{2\tau d_{B_\ell}}{\eta_{B_\ell}}\right)
$$

We also follow [G. Vardoyan and S. Wehner, 2023 IEEE Int. Conf. Quantum Comp. Eng. (QCE), vol. 01 (2023), pp. 1238–1248.] by setting utility to be the sum of the log of the rates of each link, in order to ensure weaker links are not left behind by the optimizer. This utility equation is for each source and is added up for all sources to get the total network utility.

**Network Utility:**

$$
\mathcal{U}_m = \sum_{(u_{A_\ell},u_{B_\ell})\in\mathcal{L}_m} \log_{10}\mathcal{R}_\ell
$$

Since we want to maximize the utility such that fidelity is above a fidelity threshold and that there are enough channels from the source, we can write this problem as a MINLP problem.

**Optimization Problem:**

$$
(\hat{\mu}_m,\{\hat{K}_\ell\}) = \underset{\mu_m, \{K_\ell\}}{\mathrm{argmax}}  \mathcal{U}_m
$$

**Subject to constraints:**

$$
\begin{aligned}
\mathcal{F}_\ell &\ge f_\ell \qquad
\text{and}  \qquad
\sum_{\ell} K_\ell \le K \qquad \forall(u_{A_\ell},u_{B_\ell})\in\mathcal{L}_m
\end{aligned}
$$

where $\mu_m\in(0,\infty)$ and $K_\ell\in\mathbb{N}_0$. 


In the pipeline, this MINLP problem is solved with the APOPT solver which splits the larger problem into simpler NLP subproblems. APOPT determines the frequency channel allocation very quickly, enabling the pipeline to be able to run many times.

**Frequency Scheduling**
Lastly, we use the CP-SAT solver to check if our network has interfering frequency channels. This algorithm is very efficient at scheduling due to its ability to eliminate combinations and quickly backtrack. If we are unable to find a solution, the pipeline backtracks to the next best combination we found in the routing phase.



---

## Detailed Mathematical Formulation

We summarize our complete global RSA problem formally as the following:

1. Begin with a graph consisting of $S+U$ vertices split into two types: $S$ sources $\\{s_1,\ldots,s_S\\}=V_S$ and $U$ users $\\{u_1,\ldots,u_U\\}=V_U$. These vertices are connected by $E$ undirected edges $E = \\{e_1,\ldots,e_E\\} \subseteq \\{(v,w):v,w \in V_S \cup V_U,v \neq w \\}$, each of which is characterized by an efficiency parameter $h_{e}\in(0,1]$.

2. The users wish to establish a maximum of $L\leq \lfloor U/2\rfloor$ pairwise, monogamous links described by the set $\\{(u_{1_A},u_{1_B}),\ldots,(u_{L_A},u_{L_B})\\}=L$, where each link $\ell\in[L]$ possesses an "Alice" $u_{\ell_A}$ and "Bob" $u_{\ell_B}$ and requests entanglement of fidelity exceeding some threshold $F_\ell\geq f_\ell$.

3. Each source $s_m$ ($m\in[S]$) possesses $K$ pairs of entangled frequency bins $K=\\{k_{\pm 1},\ldots,k_{\pm K}\\}$, each produced at the same flux $\mu_m$. Under this definition, $|K|=2K$. The set of links serviced by source $s_m$ is denoted by $L_m\subset L$.

4. The quantum RSA problem seeks to determine the following:
   
   a. Channel flux value $\mu_m$ for each source $m\in[S]$.
   
   b. Source assignment $\ell_S\in[S]$ and lightpaths to each user: let $P_{\ell_A}\subseteq E$ be the set of edges comprising a path from $s_{\ell_S}$ to $u_{\ell_A}$, and let $P_{\ell_B}\subseteq E$ be the set of edges comprising a path from $s_{\ell_S}$ to $u_{\ell_B}$, for each link $\ell\in[L]$. These lightpaths have efficiencies:
   
   $$\eta_{\ell_A}=\prod_{e\in P_{\ell_A}} h_e \quad \text{and} \quad \eta_{\ell_B}=\prod_{e\in P_{\ell_B}} h_e$$
   
   c. Spectral assignment for each link $\ell\in[L]$, where $K_{\ell_A}\subset K$ bins are assigned to Alice and $K_{\ell_B}\subset K$ to Bob, which are energy-matched: i.e., for every $k_n\in K_{\ell_A}$, it must be the case that $k_{-n}\in K_{\ell_B}$; hence $|K_{\ell_A}|=|K_{\ell_B}|\equiv K_\ell$.

The goal is to maximize the chosen definition of network utility $U$ subject to the following constraints:

**Constraints:**

(C1) Fidelity: $F_\ell\geq f_\ell \quad \forall \ell\in[L]$

(C2) Frequency-bin availability: $\sum_{(u_{\ell_A},u_{\ell_B})\in L_m} K_\ell \leq K \quad \forall m\in[S]$

(C3) Contention-free distribution: $K_{\ell_\alpha} \cap K_{\ell'_\beta}=\emptyset \quad \forall \ell\neq\ell', \alpha,\beta\in\{A,B\}$

where paths $P_{\ell_\alpha}$ and $P_{\ell'_\beta}$ share an edge for constraint (C3).

## Setup Instructions

This project uses Python dependencies listed in `requirements.txt`. To install them, first create and activate a virtual environment.

### 1. Create a virtual environment

Run the following command from the project directory:

**Windows:**
```bash
python -m venv env
```

**Mac/Linux:**
```bash
python3 -m venv env
```

### 2. Activate the virtual environment

**Windows Command Prompt:**
```bash
env\Scripts\activate
```

**Windows PowerShell:**
```bash
.\env\Scripts\Activate.ps1
```

If PowerShell gives an error saying that running scripts is disabled on this system, run:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Mac/Linux:**
```bash
source env/bin/activate
```

### 3. Install the required packages

After the virtual environment is activated, install the required libraries using:

**Windows:**
```bash
pip install -r requirements.txt
```

**Mac/Linux:**
```bash
python3 -m pip install -r requirements.txt
```
