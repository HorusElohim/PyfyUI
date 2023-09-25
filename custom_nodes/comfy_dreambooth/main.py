import os
import sys
import subprocess
import logging
import click
import colorlog
from pathlib import Path
import cv2
import dlib
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch

# Set up color logging
handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter("%(log_color)s%(levelname)s: %(message)s")
)
log = colorlog.getLogger(__name__)
log.addHandler(handler)
log.setLevel(logging.INFO)


def _get_venv_bin(venv_path: Path) -> Path:
    return venv_path / "Scripts" if os.name == "nt" else venv_path / "bin"


def _get_venv_pip(venv_path: Path) -> Path:
    return _get_venv_bin(venv_path) / "pip"


def _get_venv_python(venv_path: Path) -> Path:
    return _get_venv_bin(venv_path) / "python"


def create_venv(venv_path):
    try:
        log.info(f"Creating virtual environment at {venv_path}")
        if not Path(venv_path).exists():
            subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        else:
            log.info(f"Virtual environment at {venv_path}")
    except subprocess.CalledProcessError:
        log.error("Failed to create virtual environment.")
        sys.exit(1)


def install_requirements(venv_path):
    requirements_file = "requirements.txt"
    venv_pip_path = _get_venv_pip(venv_path)
    try:
        log.info(
            f"Installing requirements from {requirements_file} in virtual environment"
        )
        subprocess.run([venv_pip_path, "install", "-r", requirements_file], check=True)
    except subprocess.CalledProcessError:
        log.error("Failed to install requirements.")
        sys.exit(1)


def detect_and_crop_face(image_path, output_path, output_size=1024, scale_factor=2):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale (dlib works with grayscale images)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a face detector using dlib
    face_detector = dlib.get_frontal_face_detector()

    # Detect faces in the image
    faces = face_detector(gray_image)

    if len(faces) == 0:
        print("No faces detected in the image.")
        return

    # Get the first detected face (assuming there's only one face in the image)
    face = faces[0]

    # Calculate the coordinates for cropping
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    center_x, center_y = x + w // 2, y + h // 2
    new_size = int(max(w, h) * scale_factor)

    # Calculate the cropping region
    crop_x1 = max(center_x - new_size // 2, 0)
    crop_y1 = max(center_y - new_size // 2, 0)
    crop_x2 = min(crop_x1 + new_size, image.shape[1])
    crop_y2 = min(crop_y1 + new_size, image.shape[0])

    # Crop the image around the detected face
    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

    # Resize the cropped image to the desired output size (1024x1024)
    resized_image = cv2.resize(cropped_image, (output_size, output_size))

    # Save the cropped and resized image
    cv2.imwrite(output_path, resized_image)

    log.info(f"Face detected and cropped. Result saved to {output_path}")


def process_images_in_directory(input_path, output_path):
    log.info(f"Preprocess started")
    input_path = Path(input_path)
    output_path = Path(output_path)
    log.info(f"input: {input_path.absolute()}\toutput: {output_path.absolute()}")

    if not output_path.exists():
        output_path.mkdir(parents=True)
        log.info(f"mkdir output dir")

    for image_path in input_path.glob("*.JPG"):
        log.info(f"analyzing image: {image_path}")
        output_image_path = output_path / f"cropped_{image_path.name}"
        detect_and_crop_face(
            str(image_path),
            output_size=1024,
            scale_factor=1.3,
            output_path=str(output_image_path),
        )
        log.info(f"Processed and saved: {output_image_path}")


def test_lora(prompt: str, lora_path : Path, ):
    # Define how many steps and what % of steps to be run on each experts (80/20) here
    n_steps = 40
    high_noise_frac = 0.8
    
    model = "stabilityai/stable-diffusion-xl-base-1.0"
    base = DiffusionPipeline.from_pretrained(
        model,
        torch_dtype=torch.float16,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        use_safetensors=True,
        output_type="latent",
    )
    base.to("cuda")
    base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
    
    lora_folder = lora_path.parent
    lora_name = lora_path.name
    base.load_lora_weights(lora_folder, weight_name=lora_name)

    refiner =  DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
    )
    refiner.to("cuda")
    refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
    
    generator = torch.Generator("cuda").manual_seed(150)
    image = base(prompt=prompt, generator=generator, num_inference_steps=25, width=1024, height=1024)
    image = image.images[0]
    image.save(f"{prompt.replace(' ', '_')}.png")
    # image = refiner(prompt=prompt, generator=generator, image=image)
    # image = image.images[0]
    # image.save(f"images_refined/{seed}.png")

@click.group()
def cli():
    pass


@cli.command()
@click.argument("venv_path", type=click.Path(), default="venv")
def install(venv_path):
    create_venv(Path(venv_path))
    install_requirements(Path(venv_path))


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def preprocess(input_path, output_path):
    process_images_in_directory(input_path, output_path)

@cli.command()
@click.argument('prompt', type=str)
@click.argument('lora_path', type=click.Path(exists=True))
def testlora(prompt, lora_path):
    test_lora(prompt, Path(lora_path))

if __name__ == "__main__":
    cli()
