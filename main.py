from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from starlette.middleware.sessions import SessionMiddleware
import os
import hashlib
import cv2
import numpy as np
import logging
from PIL import Image, ImageDraw, ImageFont
import time

# ==========================================================
# ⚡ LOGGING
# ==========================================================
os.makedirs("logging", exist_ok=True)
logging.basicConfig(
    filename="logging/app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="your_secret_here")  # Change this
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==========================================================
# 🗄️ DATABASE
# ==========================================================
Base = declarative_base()
engine = create_engine("sqlite:///database.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, index=True)
    roll_no = Column(String, unique=True, index=True)
    security_word = Column(String)

class ImageData(Base):
    __tablename__ = "image_data"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    filename = Column(String)  # original filename
    output_deleted_by_user = Column(Integer, default=0)  # 0 = Not deleted, 1 = Deleted by user, 2 = Deleted by admin

Base.metadata.create_all(bind=engine)

MAX_IMAGE_LIMIT = 15
INACTIVITY_LIMIT = 600  # 10 mins

# ==========================================================
# ⚡ UTILITY FUNCTIONS
# ==========================================================
def hash_word(word: str) -> str:
    return hashlib.sha256(word.encode("utf-8")).hexdigest()

def process_image(image_path, max_width=200, character='.', detection_mode='edge',
                  target_width=None, target_height=None):
    """Convert an image to a dot-style ASCII using edge detection and optional custom scale."""
    color_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if color_img is None:
        raise ValueError(f"Unable to read the image at path: {image_path}")

    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    h, w = gray_img.shape
    aspect_ratio = h / w

    if target_width:
        new_width = target_width
        new_height = target_height if target_height else int(new_width * aspect_ratio)
    else:
        new_width = max_width
        new_height = int(new_width * aspect_ratio)

    color_img = cv2.resize(color_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    gray_img = cv2.resize(gray_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

    edges = cv2.Canny(gray_img, threshold1=100, threshold2=200)

    ascii_lines = []
    if detection_mode == "edge":
        for y in range(new_height):
            line = []
            for x in range(new_width):
                line.append(character if edges[y, x] > 0 else ' ')
            ascii_lines.append(''.join(line))
    else:
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        for y in range(new_height):
            line = []
            for x in range(new_width):
                if edges[y, x] > 0:
                    line.append(character)
                else:
                    contrast_value = abs(laplacian[y, x])
                    if contrast_value > 50:
                        line.append(character)
                    elif contrast_value > 20:
                        line.append(character if (x + y) % 2 == 0 else ' ')
                    else:
                        line.append(' ')
            ascii_lines.append(''.join(line))

    return '\n'.join(ascii_lines), color_img


def save_as_png(output_file, text_result,
                font_name='cour.ttf',
                font_size=20,
                character='.',
                color_mode='original',
                custom_dot_color='#ffffff',
                bg_color='#000000',
                color_img=None):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(BASE_DIR, font_name)

    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font file not found: {font_path}")

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        raise RuntimeError(f"Error loading font from {font_path}: {e}")

    lines = text_result.split('\n') or [" "]

    sample_char = "A"
    bbox = font.getbbox(sample_char)
    char_width = bbox[2] - bbox[0]
    line_height = bbox[3] - bbox[1]

    img_width = max(len(line) for line in lines) * char_width + 20
    img_height = len(lines) * line_height + 20

    img = Image.new("RGB", (img_width, img_height), bg_color)
    draw = ImageDraw.Draw(img)

    for y_offset, line in enumerate(lines):
        for x_offset, ch in enumerate(line):
            if ch != ' ':
                if color_mode == 'original' and color_img is not None:
                    pixel_color = tuple(color_img[y_offset, x_offset].tolist())
                elif color_mode == 'black':
                    pixel_color = "#FFFFFF"
                else:
                    pixel_color = custom_dot_color
                draw.text((10 + x_offset * char_width,
                           10 + y_offset * line_height),
                           character,
                           font=font,
                           fill=pixel_color)

    img.save(output_file)
    return True

# ==========================================================
# ⚡ ROUTES
# ==========================================================
templates = Jinja2Templates(directory="templates")

@app.get("/")
def index(request: Request):
    """Landing page."""
    if "user_id" in request.session:
        return RedirectResponse("/upload")
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/register")
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register")
def register(request: Request, roll_no: str = Form(...), security_word: str = Form(...)):
    db = SessionLocal()
    existing_user = db.query(User).filter(User.roll_no == roll_no).first()
    if existing_user:
        db.close()
        return templates.TemplateResponse("register.html",
                                          {"request": request, "error": "User already registered!"})
    user = User(roll_no=roll_no, security_word=hash_word(security_word))
    db.add(user)
    db.commit()
    db.close()
    logger.info(f"User registered: {roll_no}")
    return RedirectResponse("/login", status_code=303)


@app.get("/login")
def login(request: Request):
    """Login page."""
    if "user_id" in request.session:
        return RedirectResponse("/upload")
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
def do_login(request: Request, roll_no: str = Form(...), security_word: str = Form(...)):
    db = SessionLocal()
    user = db.query(User).filter(User.roll_no == roll_no,
                                 User.security_word == hash_word(security_word)).first()
    if user:
        request.session["user_id"] = user.id
        request.session["roll_no"] = user.roll_no
        request.session["last_active"] = time.time()
        db.close()
        logger.info(f"User logged in: {roll_no}")
        return RedirectResponse("/upload", status_code=303)

    db.close()
    logger.warning(f"Invalid login attempt for roll_no: {roll_no}")
    return templates.TemplateResponse("login.html",
                                      {"request": request, "error": "Invalid credentials!"})


@app.get("/upload")
def upload(request: Request):
    """Upload page."""
    if "user_id" not in request.session:
        return RedirectResponse("/")
    request.session["last_active"] = time.time()
    return templates.TemplateResponse("upload.html", {"request": request})


@app.get("/history")
def history(request: Request):
    """Display output-only history for logged-in user."""
    if "user_id" not in request.session:
        return RedirectResponse("/")

    user_id = request.session["user_id"]

    db = SessionLocal()
    images = db.query(ImageData).filter(ImageData.user_id == user_id).all()
    db.close()

    grouped_entries = {}
    for entry in images:
        if entry.filename not in grouped_entries or entry.id > grouped_entries[entry.filename].id:
            grouped_entries[entry.filename] = entry

    history_entries = []
    for entry in grouped_entries.values():
        output_filename = f"ascii_{entry.filename}"
        output_file = os.path.join("static/output", output_filename)

        output_exists = os.path.exists(output_file)

        if output_exists:
            output_status = None
        else:
            output_status = "user" if entry.output_deleted_by_user == 1 else ("admin" if entry.output_deleted_by_user == 2 else None)

        history_entries.append({
            "filename": entry.filename,
            "url": f"output/{output_filename}" if output_exists else None,
            "output_exists": output_exists,
            "output_status": output_status,
        })

    return templates.TemplateResponse("history.html",
                                      {"request": request,
                                       "history_entries": sorted(history_entries, key=lambda x: x["filename"], reverse=True)})


@app.post("/process")
async def process(request: Request,
                  file: UploadFile = File(...),
                  char: str = Form('.'),
                  color_mode: str = Form(...),
                  detection_mode: str = Form("edge"),
                  custom_dot_color: str = Form('#ffffff'),
                  bg_color: str = Form('#000000'),
                  output_style: str = Form('ascii'),
                  custom_scale: str = Form(None),
                  target_width: str = Form(None),
                  target_height: str = Form(None)):
    """Processes uploaded image and returns the result as ASCII."""
    if "user_id" not in request.session:
        return RedirectResponse("/")

    user_id = request.session["user_id"]
    roll_no = request.session["roll_no"]

    db = SessionLocal()
    image_count = db.query(ImageData).filter(ImageData.user_id == user_id,
                                              ImageData.output_deleted_by_user == 0).count()
    if image_count >= MAX_IMAGE_LIMIT:
        db.close()
        error_message = (
            f"Upload limit reached ({image_count}/{MAX_IMAGE_LIMIT}). "
            "Please delete old images from History to upload more."
        )
        logger.warning(error_message)
        return templates.TemplateResponse("upload.html",
                                          {"request": request, "error": error_message})
    filename = f"image_{image_count + 1}_{roll_no}.png"
    filepath = os.path.join("processed_image", filename)

    os.makedirs("processed_image", exist_ok=True)
    os.makedirs("static/output", exist_ok=True)

    with open(filepath, "wb") as f:
        f.write(await file.read())

    db.add(ImageData(user_id=user_id, filename=filename))
    db.commit()
    db.close()

    tw = int(target_width) if (custom_scale and target_width) else None
    th = int(target_height) if (custom_scale and target_height) else None

    text_result, color_img = process_image(filepath,
                                           character=char,
                                           detection_mode=detection_mode,
                                           target_width=tw,
                                           target_height=th)

    output_filename = f"ascii_image_{image_count + 1}_{roll_no}.png"
    output_path = os.path.join("static/output", output_filename)

    save_as_png(output_path, text_result,
                character=char,
                color_mode=color_mode,
                custom_dot_color=custom_dot_color,
                bg_color=bg_color,
                color_img=color_img)

    logger.info(f"User {roll_no} uploaded and processed {filename}")

    request.session["last_active"] = time.time()

    return templates.TemplateResponse("upload.html",
                                      {"request": request,
                                       "text_result": text_result,
                                       "output_png": f"/static/output/{output_filename}"})


@app.post("/process_ajax")
async def process_ajax(request: Request,
                      file: UploadFile = File(...),
                      char: str = Form('.'),
                      color_mode: str = Form('black'),
                      detection_mode: str = Form("edge"),
                      custom_dot_color: str = Form('#ffffff'),
                      bg_color: str = Form('#000000'),
                      output_style: str = Form('ascii'),
                      custom_scale: str = Form(None),
                      target_width: str = Form(None),
                      target_height: str = Form(None)):
    """Processes uploaded image and returns the result as JSON (no page refresh)."""
    if "user_id" not in request.session:
        return JSONResponse({"error": "Not logged in"}, status_code=403)

    user_id = request.session["user_id"]
    roll_no = request.session["roll_no"]

    db = SessionLocal()
    image_count = db.query(ImageData).filter(ImageData.user_id == user_id,
                                              ImageData.output_deleted_by_user == 0).count()
    if image_count >= MAX_IMAGE_LIMIT:
        db.close()
        error_message = (
            f"Upload limit reached ({image_count}/{MAX_IMAGE_LIMIT}). "
            "Please delete old images from History to upload more."
        )
        logger.warning(error_message)
        return JSONResponse({"error": error_message}, status_code=400)

    filename = f"image_{image_count + 1}_{roll_no}.png"
    filepath = os.path.join("processed_image", filename)

    os.makedirs("processed_image", exist_ok=True)
    os.makedirs("static/output", exist_ok=True)

    with open(filepath, "wb") as f:
        f.write(await file.read())

    db.add(ImageData(user_id=user_id, filename=filename))
    db.commit()
    db.close()

    tw = int(target_width) if (custom_scale and target_width) else None
    th = int(target_height) if (custom_scale and target_height) else None

    text_result, color_img = process_image(filepath,
                                           character=char,
                                           detection_mode=detection_mode,
                                           target_width=tw,
                                           target_height=th)

    output_filename = f"ascii_image_{image_count + 1}_{roll_no}.png"
    output_path = os.path.join("static/output", output_filename)

    save_as_png(output_path, text_result,
                character=char,
                color_mode=color_mode,
                custom_dot_color=custom_dot_color,
                bg_color=bg_color,
                color_img=color_img)

    logger.info(f"User {roll_no} uploaded via AJAX and processed {filename}")

    request.session["last_active"] = time.time()

    return JSONResponse({
        "text_result": text_result,
        "output_png": f"/static/output/{output_filename}"
    })


@app.get("/count_outputs")
def count_outputs(request: Request):
    """Return the number of output images a user has that are NOT deleted."""
    if "user_id" not in request.session:
        return JSONResponse({"error": "Not logged in"}, status_code=403)

    user_id = request.session["user_id"]

    db = SessionLocal()
    entries = db.query(ImageData).filter(ImageData.user_id == user_id,
                                          ImageData.output_deleted_by_user == 0).all()
    db.close()

    total_images = len(entries)

    return JSONResponse({"count": total_images, "total_images": total_images, "max_limit": MAX_IMAGE_LIMIT})


@app.post("/clear_outputs")
def clear_outputs(request: Request):
    """Delete all output images for the logged-in user and mark as deleted by user."""
    if "user_id" not in request.session:
        return JSONResponse({"error": "Not logged in"}, status_code=403)

    user_id = request.session["user_id"]

    db = SessionLocal()
    entries = db.query(ImageData).filter(ImageData.user_id == user_id).all()
    for entry in entries:
        output_filename = f"ascii_{entry.filename}"
        output_file = os.path.join("static/output", output_filename)

        if os.path.exists(output_file):
            os.remove(output_file)

        entry.output_deleted_by_user = 1
    db.commit()
    db.close()
    logger.info(f"All output images deleted for user_id {user_id}")

    return JSONResponse({"status": "all output images deleted"})


@app.get("/logout")
def logout(request: Request):
    """Logout user and delete associated original images due to inactivity."""
    if "user_id" in request.session:
        user_id = request.session["user_id"]

        db = SessionLocal()
        entries = db.query(ImageData).filter(ImageData.user_id == user_id).all()
        for entry in entries:
            original_file = os.path.join("processed_image", entry.filename)
            if os.path.exists(original_file):
                os.remove(original_file)

        db.close()
        logger.info(f"Original images deleted for user_id {user_id}")

    request.session.clear()
    return RedirectResponse("/")


@app.post("/delete_output_only")
def delete_output_only(request: Request, filename: str = Form(...)):
    """Deletes the output file for the user and marks status as deleted by user (1)."""
    if "user_id" not in request.session:
        return RedirectResponse("/")

    user_id = request.session["user_id"]

    db = SessionLocal()
    entry = db.query(ImageData).filter(ImageData.user_id == user_id,
                                        ImageData.filename == filename).order_by(ImageData.id.desc()).first()
    if entry:
        output_filename = f"ascii_{filename}"
        output_file = os.path.join("static/output", output_filename)

        if os.path.exists(output_file):  # Output present
            os.remove(output_file)

        entry.output_deleted_by_user = 1
        db.commit()
        logger.info(f"User {user_id} deleted output for {filename}")

    db.close()
    return RedirectResponse("/history", status_code=303)


@app.post("/delete_both")
def delete_both(request: Request, filename: str = Form(...)):
    """Deletes the output image for the user and marks status appropriately."""
    if "user_id" not in request.session:
        return RedirectResponse("/")

    user_id = request.session["user_id"]

    db = SessionLocal()
    entry = db.query(ImageData).filter(ImageData.user_id == user_id,
                                        ImageData.filename == filename).order_by(ImageData.id.desc()).first()
    if entry:
        output_filename = f"ascii_{filename}"
        output_file = os.path.join("static/output", output_filename)

        if os.path.exists(output_file):  # Output present
            os.remove(output_file)
            entry.output_deleted_by_user = 1
        else:
            entry.output_deleted_by_user = entry.output_deleted_by_user or 2

        db.commit()
        logger.info(f"User {user_id} deleted both files for {filename}")

    db.close()
    return RedirectResponse("/history", status_code=303)


@app.middleware("http")
async def remove_inactive_files(request: Request, call_next):
    """Check user activity and remove original files if inactive for more than 10 minutes."""
    response = await call_next(request)

    if "user_id" in request.session:
        last_active = request.session.get("last_active")
        if last_active and (time.time() - last_active) > INACTIVITY_LIMIT:
            user_id = request.session["user_id"]

            db = SessionLocal()
            entries = db.query(ImageData).filter(ImageData.user_id == user_id).all()
            for entry in entries:
                original_file = os.path.join("processed_image", entry.filename)
                if os.path.exists(original_file):
                    os.remove(original_file)

            db.close()
            logger.info(f"Original files deleted due to inactivity for user_id {user_id}")

    return response

# ==========================================================
# ⚡ MAIN ENTRYPOINT
# ==========================================================
if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)