# Arc2Face Implementation

This repository contains the code for the **Arc2Face** implementation. It includes tools for generating embeddings and reconstructing images from those embeddings. (Further plans involve creating a playground for **FaceCloak** for generating disruptors for and cloak generation.)

## Getting Started

### Prerequisites

1. **Python Version**: Ensure you have Python `3.11.10` installed.
2. **Virtual Environment**: It is recommended to create a virtual Python environment before proceeding to avoid dependency conflicts.
3. **CUDA Support**: Ensure you have a CUDA-compatible GPU and the necessary drivers installed for optimal performance.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tahmeedKENJI/Face-Encoder-Decoder-and-Security.git <repository_directory>
   cd <repository_directory>/Python_Scripting
   ```

2. Initialize the `Arc2Face` submodule:
   ```bash
   git submodule update --init --recursive
   ```

3. Create and activate a virtual environment:
   ```bash
   
   python3 -m venv venv
   source venv/bin/activate  # On Linux/Mac
   venv\Scripts\activate     # On Windows
   ```

4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Setup

1. Generate the `models` folder by running the `setup_pkg.py` script:
   ```bash
   python setup_pkg.py
   ```
2. Delete `glintr100.onnx` from `models\antelopev2`. Make sure `arc2face.onnx` is present (I missed this step. And wasted 12 whole hours wondering why are the face reconstructions so broken. I might sound like AI and annoy some people. Have fun with it. Anyway, make sure this step is enforced.)
3. Verify that the `models` folder is created successfully.

### Usage

The `default_arc2face.py` script is used to generate images based on embeddings extracted from a given input image. Follow the steps below to use the script:

1. **Prepare Input and Output Folders**:
   - Create a folder to store your input images (e.g., `input_images/`).
   - Create a folder to store the generated output images (e.g., `output_images/`).

   Example:
   ```bash
   mkdir input_images output_images
   ```

2. **Add Input Images**:
   - Place the images you want to process in the `input_images/` folder.

3. **Run the Script**:
   - Use the following command to run the script:
     ```bash
     python default_arc2face.py <path_to_input_image> <path_to_output_folder> <num_of_images_to_generate>
     ```
   - Replace `<path_to_input_image>` with the relative or absolute path to the input image.
   - Replace `<path_to_output_folder>` with the relative or absolute path to the folder where the generated images will be saved.
   - Replace `<num_of_images_to_generate>` with the number of required output images

   Example:
   ```bash
   python default_arc2face.py input_images/sample.jpg output_images 5
   ```

4. **Output**:
   - The script will generate and save the images in the specified output folder with filenames like `gen_image_0.jpg`, `gen_image_1.jpg`, etc.

### Directory Structure

```
Python_Scripting/
├── README.md
├── requirements.txt
├── setup_pkg.py
├── default_arc2face.py
├── input_images/  # Folder for input images (create manually)
├── output_images/ # Folder for output images (create manually)
└── models/        # Generated after running setup_pkg.py
```

### Notes

- Ensure that the input image is a valid image file (e.g., `.jpg`, `.png`).
- The script automatically detects the largest face in the input image for processing.
- You can modify the `num_images` parameter in the script to generate more or fewer images.

### Troubleshooting

- **Missing Dependencies**: Ensure that the virtual environment is activated and all packages from `requirements.txt` are installed.
- **CUDA Issues**: Verify that your GPU drivers and CUDA toolkit are properly installed.
- **Submodule Issues**: Ensure the submodule is initialized and updated correctly using:
  ```bash
  git submodule update --init --recursive
  ```

### Upcoming Updates

Currently, the project only contains the encoder-decoder pipeline. No implementations of FaceCloak is available yet. In future iteration, hopefully, these updates will see the light of day. README.md file will be updated accordingly.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- This project uses
[Arc2Face](https://huggingface.co/FoivosPar/Arc2Face) hugging-face and [Arc2Face](https://github.com/foivospar/Arc2Face) github submodule
[FaceCloak](https://github.com/sudban3089/FaceCloak) github submodule
[InsightFace](https://github.com/deepinsight/insightface) library.
- Special thanks to contributors and open-source libraries that made this project possible.

---

## Contact

If you have any questions, suggestions, or issues, feel free to reach out:

- **Name**: S M Tahmeed Reza
- **Email**: tahmeedreza@gmail.com
- **GitHub**: [tahmeedKENJI](https://github.com/tahmeedKENJI)
- **LinkedIn**: [S. M. Tahmeed Reza](https://www.linkedin.com/in/s-m-tahmeed-reza-1b1870322)

---
