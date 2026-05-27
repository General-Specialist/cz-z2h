Lecture 1: Representing 3D Biological Space

  - The Goal: Transition from raw molecular files (.cif) and cryo-EM maps
    (.mrc) into raw PyTorch tensors without complex dependencies.
  - The Concepts:
      - Parsing physical coordinates (x, y, z) from text files.
      - Representing 3D space as a voxelized grid (density map).
      - Rasterization: Writing a simple custom function to turn discrete
        coordinates into a continuous 3D density grid with a Gaussian blur
        kernel to simulate a synthetic cryo-EM density map.
  - From-Scratch Code: A raw Python PDB parser and a custom grid-rasterization
    function using only PyTorch tensors.

Lecture 2: Feature Extraction in 3D (The 3D U-Net)

  - The Goal: Build a neural network that processes a raw 3D density map to
    predict where atoms are located.
  - The Concepts:
      - 3D Convolutions: understanding strides, padding, and the steep memory
        cost of 3D grids.
      - The U-Net architecture: Downsampling (pooling), upsampling (transposed
        convolutions), and skip connections.
      - Extracting "support points": Writing a simple 3D peak-finding algorithm
        (Non-Maximum Suppression) to turn the U-Net’s continuous density
        predictions into discrete coordinate points.
  - From-Scratch Code: A complete 3D U-Net module in PyTorch and a
    clustering/peak-finding layer. We will train it to locate "carbon atoms" in
    a toy synthetic density map.

Lecture 3: Representing the Sequence (The Pairformer)

  - The Goal: Represent sequences (proteins and nucleic acids) and build the
    framework that predicts how residues interact.
  - The Concepts:
      - Representing 1D sequences as embeddings.
      - Constructing a 2D "pair representation" matrix
        (N_{\text{res}} \times N_{\text{res}}) representing relative distances
        and spatial relationships.
      - Implementing a lightweight Pairformer (a core block of AlphaFold 3 /
        Protenix). We’ll write self-attention over sequence residues and show
        how the 2D matrix acts as a map of potential physical contacts.
  - From-Scratch Code: Sequence-to-pair embedding layers, multi-head attention
    on sequences, and a simplified Pairformer block.

Lecture 4: Bridging 1D and 3D (The EM-Pairformer)

  - The Goal: Connect the 1D biological sequence with the physical 3D density
    map. This is the heart of CryoZeta’s innovation.
  - The Concepts:
      - We have sequence residues (N) and spatial support points extracted from
        the map (M). How do they talk?
      - The 4 joint representations of CryoZeta: single sequence,
        residue-residue pair, support point-residue pair, and support
        point-support point.
      - Cross-Attention: Writing the mechanism where sequence residues "attend"
        to physical 3D coordinates, updating their representations based on
        nearby electron density.
  - From-Scratch Code: The EM-Pairformer block. Students will inspect the
    attention maps to see exactly how a residue (e.g., Alanine) locates its
    corresponding physical spot in the density grid.

Lecture 5: 3D Coordinate Generation via Diffusion (The Structure Module)

  - The Goal: Generate full-atom 3D coordinates directly from our neural
    representations.
  - The Concepts:
      - Continuous Diffusion on 3D point clouds.
      - How noise is added to atomic coordinates, and how a neural network is
        trained to predict the noise (or the denoised coordinate) at each
        timestep t.
      - Conditioning: How sequence embeddings guide the diffusion model to
        assemble a coherent structure rather than an arbitrary point cloud.
  - From-Scratch Code: A mini 3D coordinate diffusion module. We will train it
    on a toy dataset to denoise random 3D coordinates into a structured shape
    (like a simple alpha-helix) conditioned on a mock sequence embedding.

Lecture 6: Loss Functions and Fitting Constraints

  - The Goal: Keep the predicted structure physically plausible and aligned with
    the cryo-EM map.
  - The Concepts:
      - Why standard Mean Squared Error (MSE) is poor for rotating structures
        (the alignment problem).
      - Frame Aligned Point Error (FAPE) or rigid alignment loss: calculating
        error in local coordinate frames.
      - The Distogram Head: Predicting distances between sequence residues and
        density support points to act as a secondary fitting constraint.
  - From-Scratch Code: Implementing a custom FAPE loss function and a coordinate
    superposition step (solving the Kabsch algorithm using SVD in PyTorch) to
    align our generated structure to the target map.

Lecture 7: The Grand Finale (End-to-End Training)

  - The Goal: Connect all components into a mini-CryoZeta model and train it on
    a single target.
  - The Scenario:
      - We will use a tiny, well-characterized protein complex (e.g., a small
        dimer) with a simulated low-resolution cryo-EM map (e.g., 8 Å
        resolution).
      - We will pipe the sequence and map through our U-Net, feed the support
        points and sequence to the EM-Pairformer, run the coordinate diffusion
        denoising process, and apply our FAPE and distance-guidance losses.
      - We will write a custom training loop, print training loss, and watch the
        model assemble a scrambled set of atoms into the physical shape of the
        electron density map in real-time.
  - From-Scratch Code: A unified script/notebook compiling the entire pipeline,
    with a visualization step using basic plotting or py3Dmol to render the
    folded output directly in the Jupyter Notebook.

Pedagogical Nuances (The "Karpathy Touch")

To make this course truly "z2h-style," we would emphasize several recurring
themes:

1.  "Shape-Watching": At every step of the notebook, we would print the shapes
    of our tensors. Students would constantly see how a sequence tensor of
    [Batch, Seq_Len, Hidden_Dim] interacts with a pair representation of [Batch,
    Seq_Len, Seq_Len, Pair_Dim] and a coordinate representation of [Batch,
    Num_Atoms, 3].
2.  Handling Rigids and Rotations: A major hurdle in structural biology is 3D
    rotation. We would spend a significant portion of a lecture demystifying
    rotations—writing custom rotation matrix multiplications from scratch,
    rather than importing them, to build mathematical intuition.
3.  Debugging the Grid: We would write visual debuggers to slice through 3D
    density maps. Students would plot 2D slices of the 3D grid alongside the
    predicted support points, grounding the abstract machine learning math in
    physical reality.
4.  Humble, iterative progress: We would intentionally start with broken
    baselines (e.g., a model that collapses all atoms to the origin, or fits
    sequences to the wrong density blobs) to teach how to diagnose, debug, and
    fix common structural biology ML errors.
