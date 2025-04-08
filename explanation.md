Below is a **high-level, intuitive** explanation of why algorithms like **Difference Target Propagation (DTP)**, **Contrastive Hebbian Learning**, and **Helmholtz Machines** can approximate backpropagation by relying on *local* mechanisms and “difference” signals rather than a strict global chain rule. I will focus on conceptual motivation with minimal math.

---

## 1. The Core Challenge: Local Credit Assignment

In neural networks, you have many layers of neurons (or units). When you do **gradient backpropagation**, you compute how changing each unit’s activation or weight would affect the final loss. But in a biological (or more decentralized) system, it’s **not obvious** how each local neuron or synapse can know its own contribution to the global error. That’s the **credit assignment** problem.

- **Backprop’s solution**: We do a single global pass backward, using partial derivatives (the chain rule), which essentially tells every weight how the final error depends on it. 
- **Biological plausibility challenge**: It’s unclear if the brain (or a local learning rule) can literally perform chain rule calculations and broadcast them backward with precisely matched weights.

Hence, many alternative algorithms have been proposed—like **DTP**, **Contrastive Hebbian**, or **Helmholtz Machines**—that attempt to do something *like* backprop but via local, biologically inspired mechanisms.

---

## 2. The Key Idea: “Desired Activations” vs. “Actual Activations”

A unifying theme among these methods is: 
> **Instead of sending back explicit gradients, we provide each neuron (or each layer) with *target activations* or *desired states* that help reduce the global error.**

Then each neuron can locally adjust its weights to move its actual activation toward that target. 

### 2.1. Activations vs. Gradients

- In **backprop**, each neuron sees a gradient signal telling it “you should go up or down by this amount” (plus chain rule factors).
- In **target-based** methods (like DTP or Helmholtz):
  - The neuron sees “you should try to be *this* activation instead of your current one.”  

This approach often leverages **inference** or **feedback** connections that convey what the layer or neuron *should be* if the network were to fix the global error.

---

## 3. “Difference” as a Local Correction

### 3.1. Example in DTP

In **Difference Target Propagation**, each hidden layer receives a *difference correction*:
$$

\boxed{ 
h_i^* = \underbrace{h_i}_{\text{actual}} 
- \underbrace{g_i(h_{i+1})}_{\text{estimated cause of actual}} 
+ \underbrace{g_i(h_{i+1}^*)}_{\text{estimated cause of desired}} 
}

$$
In words:
1. $h_i$ is what the layer *did* produce.
2. $g_i(h_{i+1})$ is the “inverse guess” of what led to that next-layer activity $h_{i+1}$.
3. $g_i(h_{i+1}^*)$ is the “inverse guess” of what the next layer *should be*.

So the layer says: “I’ll correct my current activation by the difference between the *inferred cause of the desired next-layer activity* and the *inferred cause of the actual next-layer activity*.”  

This difference-based signal is effectively a local “error” measure that depends on the forward and backward (inverse) mappings but does **not** require a global chain rule pass.

### 3.2. Contrastive Hebbian & Helmholtz

- **Contrastive Hebbian** (“energy-based” or Hopfield-like networks) similarly compares two phases:
  1. **Free phase**: Let the network settle into some “actual” state given inputs.  
  2. **Clamped phase**: Force the network to produce (or partially produce) the desired output, then settle again.  
  3. **Difference**: Compare these two states and adjust weights to reduce the discrepancy.  

  That difference in states between the free and clamped phases **contains** the gradient information needed for learning—again, *locally* measured at each synapse.

- **Helmholtz Machine** uses a “wake” phase (observe data, pass it forward) and a “sleep” phase (generate from the model, pass backward), or sometimes a “wake-sleep” approach. The difference between how the network encodes real data vs. how it encodes its own “fantasy” states provides a local learning signal.

In all these schemes, the **difference** between an *actual* state and a *desired* or *constrained* state creates a local error or mismatch. By adjusting weights to reduce that mismatch, you approximate the changes you’d get from standard backprop.

---

## 4. Local Inference or Feedback Models

Because these algorithms rely on differences between “what is” and “what should be” at each layer, they each have some way of **computing or guessing** what “should be.” Often this is done with:

1. **Feedback connections** (like approximate inverse networks in DTP).
2. **Recurrent relaxation** (like in Contrastive Hebbian, where the network runs in two different phases).
3. **Generative model** (like in Helmholtz Machines, which have top-down pathways to generate data-like patterns).

The local learning rule then uses:

\[
(\text{Local mismatch}) = (\text{Actual activation}) \;-\; (\text{Desired activation})
\]

or some variant. And that mismatch *implicitly* encodes the gradient from backprop—*provided* those feedback or generative mappings are well learned. 

Hence, **the better the feedback or generative model, the closer you approximate the backprop gradient**.

---

## 5. Why Differences Approximate Gradients

In a more **mathematical** sense (still lightly):

- **Backprop** says: \( \Delta w \sim \frac{\partial \mathcal{L}}{\partial w} \).  
- **Target-based** methods say: \( \Delta w \sim (h_i^* - h_i) \times \frac{\partial h_i}{\partial w} \).

If \(h_i^*\) is chosen so that \(h_i^* - h_i\) *aligns* with the partial derivative \(\frac{\partial \mathcal{L}}{\partial h_i}\), then you effectively get the same updates as backprop.  

In DTP, for instance, 
\[
h_i^* - h_i 
\approx 
- \alpha \frac{\partial \mathcal{L}}{\partial h_i}
\]
when the inverses are good enough. This means the difference \((h_i^* - h_i)\) becomes analogous to the backprop gradient signal for \(h_i\). 

**Contrastive Hebbian** does something similar with “free phase” vs. “clamped phase” activities, and **Helmholtz Machines** with “wake” vs. “sleep” states. In each case, a difference of states approximates the gradient direction.

---

## 6. Putting It All Together

1. **Motivation**: 
   - Biological or local learning rules can’t easily do global chain rule computations.
   - Instead, let each layer or neuron see a difference between its *actual* and *desired* activity.

2. **Mechanism**: 
   - Provide local feedback or generate an *alternate* network state that represents “how the network *should* behave” if it is to reduce the final error.
   - Then the *difference* (actual vs. desired) at each neuron becomes a local error signal.

3. **Approximation to Backprop**: 
   - Mathematically, these difference signals can be shown to align with the true gradient signals, *if* the inverse or generative pathways are sufficiently trained.  
   - The better the feedback/generative model, the closer you get to standard gradient-based updates.

4. **Key Benefit**: 
   - Learning becomes more local (though not entirely free of global constraints).
   - Weight symmetry is relaxed compared to standard backprop.

5. **Common Thread**: 
   - **DTP**: local difference targets for each layer using approximate inverses.  
   - **Contrastive Hebbian**: difference in neuron activities between free and clamped phases.  
   - **Helmholtz Machines**: difference in recognition vs. generative phases (wake-sleep).  

In each, you’re effectively providing local “contrastive” signals (the difference between actual and desired states). This cleverly encodes the information that backprop’s chain rule normally carries—*but via local dynamics or local inverse models instead of a single global backward pass*.

---

### Final Takeaway

**Difference-based algorithms** (DTP, Contrastive Hebbian, Helmholtz Machines) are all ways of achieving *approximately the same effect as backprop*—that is, adjusting weights so as to minimize a global error—**by using local signals** derived from differences in actual vs. desired states. These local difference signals act as stand-ins for the partial derivatives that backprop would have computed. The entire trick relies on some auxiliary mechanism (inverse networks, clamped states, generative models) that can convey the “desired” activity or a “contrastive” state to each local unit, enabling them to do their own credit assignment with fewer global constraints.