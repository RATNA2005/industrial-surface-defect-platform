"""
McCulloch-Pitts (M-P) Neuron Implementation from Scratch
=========================================================
Historical Model of an Artificial Neuron (McCulloch & Pitts, 1943).

Demonstrates:
  - Binary inputs and outputs: x_i, y in {0, 1}
  - Excitatory weights: w_i = 1
  - Weighted summation: z = sum(w_i * x_i)
  - Threshold activation (Heaviside step function): y = 1 if z >= theta else 0
  - Canonical Logic Gates: AND, OR, NOT, AND-NOT
  - Industrial Manufacturing Defect Trigger Analogy (Conservative vs Strict Rejection)
"""

class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        """
        Initializes an M-P neuron.
        :param weights: List of synaptic weights (typically +1 for excitatory, -1 for inhibitory)
        :param threshold: Firing threshold theta
        """
        self.weights = list(weights)
        self.threshold = threshold

    def forward(self, inputs):
        """
        Computes neuron activation:
          z = sum(w_i * x_i)
          y = 1 if z >= theta else 0
        """
        if len(inputs) != len(self.weights):
            raise ValueError(f"Input dimension {len(inputs)} does not match weight dimension {len(self.weights)}")

        # Step 1: Weighted sum (linear combination)
        z = sum(w * x for w, x in zip(self.weights, inputs))

        # Step 2: Step activation against threshold theta
        y = 1 if z >= self.threshold else 0
        return y, z

def print_truth_table(gate_name, neuron, inputs_list):
    print(f"\n--- {gate_name} Gate (Weights: {neuron.weights}, Threshold theta: {neuron.threshold}) ---")
    header = " | ".join([f"x_{i+1}" for i in range(len(neuron.weights))]) + " | Weighted Sum (z) | Output (y)"
    print(header)
    print("-" * len(header))
    
    results = []
    for x in inputs_list:
        y, z = neuron.forward(x)
        inputs_str = " | ".join(str(val) for val in x)
        print(f" {inputs_str}   |        {z:2d}        |     {y}")
        results.append((x, z, y))
    return results

def run_demonstrations():
    print("=" * 65)
    print("  MCCULLOCH-PITTS NEURON: LOGIC GATES & INDUSTRIAL ANALOGY")
    print("=" * 65)

    standard_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

    # 1. AND Gate: Fires only when both inputs are active (weights=[1, 1], threshold=2)
    and_neuron = McCullochPittsNeuron(weights=[1, 1], threshold=2)
    and_res = print_truth_table("AND", and_neuron, standard_inputs)
    assert [y for _, _, y in and_res] == [0, 0, 0, 1], "AND gate verification failed!"
    print("[PASSED] AND Gate verified.")

    # 2. OR Gate: Fires when at least one input is active (weights=[1, 1], threshold=1)
    or_neuron = McCullochPittsNeuron(weights=[1, 1], threshold=1)
    or_res = print_truth_table("OR", or_neuron, standard_inputs)
    assert [y for _, _, y in or_res] == [0, 1, 1, 1], "OR gate verification failed!"
    print("[PASSED] OR Gate verified.")

    # 3. AND-NOT Gate (Inhibition): Input 1 is excitatory, Input 2 is inhibitory (weights=[1, -1], threshold=1)
    and_not_neuron = McCullochPittsNeuron(weights=[1, -1], threshold=1)
    and_not_res = print_truth_table("AND-NOT (Inhibition)", and_not_neuron, standard_inputs)
    assert [y for _, _, y in and_not_res] == [0, 0, 1, 0], "AND-NOT gate verification failed!"
    print("[PASSED] AND-NOT (Inhibitory) Gate verified.")

    # 4. Industrial Surface Defect Inspection Analogy
    print("\n" + "=" * 65)
    print("  INDUSTRIAL QUALITY DECISION ANALOGY: DUAL SENSOR COIL INSPECTION")
    print("=" * 65)
    print("Scenario: A hot-rolled steel strip passes under two optical inspection sensors:")
    print("  - Sensor 1 (x_1): Specular reflectance anomaly (e.g. surface scratches)")
    print("  - Sensor 2 (x_2): Dark-field scatter anomaly (e.g. rolled-in scale)")
    
    # Conservative Line Inspection Mode (OR Gate, theta=1)
    # High sensitivity: flag coil for human review if EITHER sensor sees a defect.
    conservative_flag = McCullochPittsNeuron(weights=[1, 1], threshold=1)
    print("\n[Policy A: Conservative QA Review Trigger (OR Gate, theta=1)]")
    for x in standard_inputs:
        y, z = conservative_flag.forward(x)
        status = "FLAG FOR REVIEW" if y == 1 else "PASS"
        print(f"  Sensor 1: {x[0]} | Sensor 2: {x[1]} -> Decision: {status}")

    # Strict Automatic Reject Mode (AND Gate, theta=2)
    # High confidence: trigger automated line rejection only when BOTH sensors agree.
    strict_reject = McCullochPittsNeuron(weights=[1, 1], threshold=2)
    print("\n[Policy B: Automated High-Confidence Reject Trigger (AND Gate, theta=2)]")
    for x in standard_inputs:
        y, z = strict_reject.forward(x)
        status = "AUTOMATIC REJECT" if y == 1 else "HOLD / RE-INSPECT"
        print(f"  Sensor 1: {x[0]} | Sensor 2: {x[1]} -> Decision: {status}")

    print("\n" + "=" * 65)
    print("  ALL MCCULLOCH-PITTS DEMONSTRATIONS PASSED [100%]")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_demonstrations()
