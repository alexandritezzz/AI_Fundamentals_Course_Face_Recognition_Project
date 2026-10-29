import argparse
import base64
import csv
import os
import shutil
import socket
import tempfile
import webbrowser
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from Controller import controller
from Recognize import Face_Recognize, Load


Application = FastAPI(title="Face Recognition Demo")

Application.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Base = Path(__file__).resolve().parent
Project_Root = Base.parent
Frontend_Path = Base / "Frontend.html"

demo_control = controller()
Is_Customized_Dataset = demo_control.Is_Customized_Dataset
Data_Path = demo_control.getDataPath()
Register_Path = demo_control.getRegPath()


def load_feature_data():
    try:
        data = Load(Data_Path)
        if data is None:
            raise RuntimeError(f"Feature database not found: {Data_Path}")
        return data
    except Exception as exc:
        print(f"\nCould not load feature database: {exc}")
        print("Trying to rebuild the database with Build.py ...\n")

        import Build

        Build.Build(Register_Path, Is_Customized_Dataset)
        data = Load(Data_Path)
        if data is None:
            raise RuntimeError(
                f"Could not load feature database {Data_Path} after rebuild."
            )
        return data


Data = load_feature_data()
Identity_Map = {}
CSV_Path = Project_Root / "dataset" / "identities.csv"

if CSV_Path.exists():
    try:
        with open(CSV_Path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    identity = row[0].strip()
                    name = row[1].strip()
                    field = row[2].strip() if len(row) > 2 else ""
                    Identity_Map[identity] = f"{name} [{field}]" if field else name
        print(f"\nLoaded identities.csv: {len(Identity_Map)} identities.\n")
    except Exception as exc:
        print(f"\nFailed to load identities.csv: {exc}")
else:
    print(f"\nWarning: {CSV_Path} was not found. Identity IDs will be displayed.\n")


def Get_Identity_Name(Identity: str) -> str:
    return Identity_Map.get(Identity, Identity)


def Get_Reference_Image(Identity: str) -> Optional[str]:
    folder_name = Identity if Is_Customized_Dataset else f"identity_{Identity}"
    folder_path = Path(Register_Path) / folder_name

    if not folder_path.exists():
        return None

    images = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(folder_path.glob(pattern))

    if not images:
        return None

    with open(images[0], "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode("utf-8")
        suffix = images[0].suffix.lower().lstrip(".")
        mime = "jpeg" if suffix == "jpg" else suffix
        return f"data:image/{mime};base64,{encoded_image}"


@Application.get("/")
async def Frontend():
    return FileResponse(Frontend_Path)


@Application.get("/favicon.ico", include_in_schema=False)
async def Favicon():
    return Response(status_code=204)


@Application.get("/health")
async def Health():
    return {"Status.": "OK"}


@Application.post("/recognize")
async def Recognize_API(Image: UploadFile = File(...)):
    suffix = Path(Image.filename or "").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temporary_path = Path(temp_file.name)
        shutil.copyfileobj(Image.file, temp_file)

    try:
        tuples = Face_Recognize(str(temporary_path), Data, Threshold=0.25)
    finally:
        temporary_path.unlink(missing_ok=True)
        await Image.close()

    results = []
    for identity, score, bbox in tuples:
        reference_encoded_image = (
            Get_Reference_Image(identity) if identity != "Unknown." else None
        )

        display_identity = (
            Get_Identity_Name(identity) if identity != "Unknown." else identity
        )

        results.append(
            {
                "Identity.": display_identity,
                "Score.": float(score),
                "BBox.": bbox or [0, 0, 0, 0],
                "Reference_Encoded_Image.": reference_encoded_image,
            }
        )

    return {"Faces.": results, "Status.": "Success"}


def get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the face recognition web demo.")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")))
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    local_url = f"http://127.0.0.1:{args.port}/"
    lan_url = f"http://{get_lan_ip()}:{args.port}/"

    print("\nFace recognition demo is running.")
    print(f"Local:   {local_url}")
    print(f"Network: {lan_url}")
    print("Press Ctrl + C to stop.\n")

    if not args.no_browser:
        webbrowser.open(local_url)

    uvicorn.run(Application, host=args.host, port=args.port)
