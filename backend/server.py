import os
import sys
import uuid
import base64
import torch
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
import numpy as np
import cv2
from PIL import Image

# Load environment variables
load_dotenv()

# Fix for PyTorch 2.6+ weights_only issue
torch.serialization.add_safe_globals([])

# Add src path for YOLO models
SRC_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

# Set environment variable to allow unsafe loading for YOLO models
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

from ultralytics import YOLO

# Initialize FastAPI
app = FastAPI(
    title="AutoDamageID API",
    description="Araç Hasar Tespit ve Analiz API'si",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017/autodamageid")
client = MongoClient(MONGO_URL)
db = client.autodamageid
analyses_collection = db.analyses

# Model paths
YOLO_DIR = Path(__file__).parent.parent / "src" / "yolo"
DAMAGE_MODEL_PATH = YOLO_DIR / "weights" / "best.pt"
PARTS_MODEL_PATH = YOLO_DIR / "runs" / "carparts_seg_v1" / "weights" / "best.pt"

# Load models (lazy loading)
damage_model = None
parts_model = None

def get_damage_model():
    global damage_model
    if damage_model is None:
        print(f"Loading damage model from {DAMAGE_MODEL_PATH}")
        # Use weights_only=False for YOLO custom models
        import torch
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        damage_model = YOLO(str(DAMAGE_MODEL_PATH))
        torch.load = original_load
    return damage_model

def get_parts_model():
    global parts_model
    if parts_model is None:
        print(f"Loading parts model from {PARTS_MODEL_PATH}")
        # Use weights_only=False for YOLO custom models
        import torch
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        parts_model = YOLO(str(PARTS_MODEL_PATH))
        torch.load = original_load
    return parts_model

# Damage type translations
DAMAGE_TR = {
    "crack": "Çatlak",
    "dent": "Göçük",
    "glass_shatter": "Cam Kırığı",
    "lamp_broken": "Lamba Kırığı",
    "scratch": "Çizik",
    "tire_flat": "Patlak Lastik"
}

# Part name translations
PARTS_TR = {
    "back_bumper": "Arka Tampon",
    "back_door": "Arka Kapı",
    "back_glass": "Arka Cam",
    "back_left_door": "Arka Sol Kapı",
    "back_left_light": "Arka Sol Far",
    "back_light": "Arka Far",
    "back_right_door": "Arka Sağ Kapı",
    "back_right_light": "Arka Sağ Far",
    "front_bumper": "Ön Tampon",
    "front_door": "Ön Kapı",
    "front_glass": "Ön Cam",
    "front_left_door": "Ön Sol Kapı",
    "front_left_light": "Ön Sol Far",
    "front_light": "Ön Far",
    "front_right_door": "Ön Sağ Kapı",
    "front_right_light": "Ön Sağ Far",
    "hood": "Kaput",
    "left_mirror": "Sol Ayna",
    "object": "Nesne",
    "right_mirror": "Sağ Ayna",
    "tailgate": "Bagaj Kapağı",
    "trunk": "Bagaj",
    "wheel": "Tekerlek"
}

# Severity mapping based on damage type
SEVERITY_MAP = {
    "crack": 3,
    "dent": 3,
    "glass_shatter": 5,
    "lamp_broken": 4,
    "scratch": 2,
    "tire_flat": 4
}

def box_iou(box_a, box_b):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    
    area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    
    union = area_a + area_b - inter_area + 1e-6
    return inter_area / union

def analyze_image(image_np: np.ndarray) -> Dict[str, Any]:
    """Run damage detection and parts segmentation on image"""
    
    damage_mod = get_damage_model()
    parts_mod = get_parts_model()
    
    h, w = image_np.shape[:2]
    
    # Run damage detection
    damage_results = damage_mod.predict(
        source=image_np,
        imgsz=640,
        conf=0.05,
        verbose=False
    )[0]
    
    # Run parts segmentation
    parts_results = parts_mod.predict(
        source=image_np,
        imgsz=640,
        conf=0.05,
        verbose=False
    )[0]
    
    # Extract damage boxes
    dmg_boxes = damage_results.boxes.xyxy.cpu().numpy() if damage_results.boxes is not None else np.zeros((0, 4))
    dmg_cls = damage_results.boxes.cls.cpu().numpy().astype(int) if damage_results.boxes is not None else np.zeros((0,), int)
    dmg_conf = damage_results.boxes.conf.cpu().numpy() if damage_results.boxes is not None else np.zeros((0,))
    dmg_names = damage_mod.names
    
    # Extract part boxes
    part_boxes = parts_results.boxes.xyxy.cpu().numpy() if parts_results.boxes is not None else np.zeros((0, 4))
    part_cls = parts_results.boxes.cls.cpu().numpy().astype(int) if parts_results.boxes is not None else np.zeros((0,), int)
    part_names = parts_mod.names
    
    # Match damages to parts
    damages = []
    for i, dmg_box in enumerate(dmg_boxes):
        best_iou = 0.0
        best_part = None
        best_part_box = None
        
        for j, part_box in enumerate(part_boxes):
            iou = box_iou(dmg_box, part_box)
            if iou > best_iou:
                best_iou = iou
                best_part = part_names[int(part_cls[j])]
                best_part_box = part_box.tolist()
        
        damage_type = dmg_names[int(dmg_cls[i])]
        confidence = float(dmg_conf[i])
        
        damage_entry = {
            "id": str(uuid.uuid4())[:8],
            "type": damage_type,
            "type_tr": DAMAGE_TR.get(damage_type, damage_type),
            "confidence": round(confidence * 100, 1),
            "severity": SEVERITY_MAP.get(damage_type, 3),
            "box": dmg_box.tolist(),
            "part": best_part if best_iou > 0.1 else None,
            "part_tr": PARTS_TR.get(best_part, best_part) if best_iou > 0.1 else None,
            "part_box": best_part_box if best_iou > 0.1 else None,
            "iou_with_part": round(best_iou, 3)
        }
        damages.append(damage_entry)
    
    # Extract unique parts detected
    parts = []
    for j, part_box in enumerate(part_boxes):
        part_name = part_names[int(part_cls[j])]
        parts.append({
            "name": part_name,
            "name_tr": PARTS_TR.get(part_name, part_name),
            "box": part_box.tolist()
        })
    
    # Calculate summary
    total_damages = len(damages)
    affected_parts = len(set([d["part"] for d in damages if d["part"]]))
    avg_severity = round(sum([d["severity"] for d in damages]) / max(1, total_damages), 1)
    
    risk_level = "Düşük"
    if avg_severity >= 4 or total_damages >= 4:
        risk_level = "Yüksek"
    elif avg_severity >= 2.5 or total_damages >= 2:
        risk_level = "Orta"
    
    return {
        "damages": damages,
        "parts": parts,
        "summary": {
            "total_damages": total_damages,
            "affected_parts": affected_parts,
            "average_severity": avg_severity,
            "risk_level": risk_level
        },
        "image_size": {"width": w, "height": h}
    }

# Pydantic models
class AnalysisResponse(BaseModel):
    id: str
    created_at: str
    image_base64: str
    results: Dict[str, Any]

class AnalysisListItem(BaseModel):
    id: str
    created_at: str
    thumbnail: str
    summary: Dict[str, Any]

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "AutoDamageID"}

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_vehicle(file: UploadFile = File(...)):
    """Upload and analyze a vehicle image for damage detection"""
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Sadece resim dosyaları kabul edilir")
    
    # Read image
    contents = await file.read()
    
    # Convert to numpy array
    nparr = np.frombuffer(contents, np.uint8)
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image_np is None:
        raise HTTPException(status_code=400, detail="Resim okunamadı")
    
    # Analyze
    results = analyze_image(image_np)
    
    # Convert image to base64 for storage/display
    _, buffer = cv2.imencode('.jpg', image_np, [cv2.IMWRITE_JPEG_QUALITY, 85])
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Create thumbnail
    thumb_size = 200
    h, w = image_np.shape[:2]
    scale = thumb_size / max(h, w)
    thumb = cv2.resize(image_np, (int(w * scale), int(h * scale)))
    _, thumb_buffer = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 60])
    thumbnail_base64 = base64.b64encode(thumb_buffer).decode('utf-8')
    
    # Create analysis record
    analysis_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    
    analysis_doc = {
        "_id": analysis_id,
        "created_at": created_at,
        "image_base64": image_base64,
        "thumbnail": thumbnail_base64,
        "results": results,
        "filename": file.filename
    }
    
    # Save to MongoDB
    analyses_collection.insert_one(analysis_doc)
    
    return AnalysisResponse(
        id=analysis_id,
        created_at=created_at,
        image_base64=image_base64,
        results=results
    )

@app.get("/api/analyses")
async def get_analyses(limit: int = 20):
    """Get list of past analyses"""
    analyses = list(analyses_collection.find().sort("created_at", -1).limit(limit))
    
    return [
        {
            "id": str(a["_id"]),
            "created_at": a["created_at"],
            "thumbnail": a.get("thumbnail", ""),
            "summary": a["results"]["summary"],
            "filename": a.get("filename", "Bilinmeyen")
        }
        for a in analyses
    ]

@app.get("/api/analyses/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get a specific analysis by ID"""
    analysis = analyses_collection.find_one({"_id": analysis_id})
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analiz bulunamadı")
    
    return {
        "id": str(analysis["_id"]),
        "created_at": analysis["created_at"],
        "image_base64": analysis["image_base64"],
        "results": analysis["results"],
        "filename": analysis.get("filename", "Bilinmeyen")
    }

@app.delete("/api/analyses/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """Delete an analysis"""
    result = analyses_collection.delete_one({"_id": analysis_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Analiz bulunamadı")
    
    return {"message": "Analiz silindi"}

@app.get("/api/analyses/{analysis_id}/pdf")
async def download_pdf(analysis_id: str):
    """Generate and download PDF report"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
    analysis = analyses_collection.find_one({"_id": analysis_id})
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analiz bulunamadı")
    
    # Create PDF in memory
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5*cm, bottomMargin=1.5*cm)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24, spaceAfter=20)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, spaceAfter=10, spaceBefore=15)
    normal_style = styles['Normal']
    
    elements = []
    
    # Title
    elements.append(Paragraph("Araç Hasar Analiz Raporu", title_style))
    elements.append(Paragraph(f"Tarih: {analysis['created_at'][:10]}", normal_style))
    elements.append(Paragraph(f"Rapor ID: {analysis_id[:8]}", normal_style))
    elements.append(Spacer(1, 20))
    
    # Add image
    img_data = base64.b64decode(analysis['image_base64'])
    img_buffer = BytesIO(img_data)
    img = RLImage(img_buffer, width=14*cm, height=10*cm, kind='proportional')
    elements.append(img)
    elements.append(Spacer(1, 20))
    
    # Summary
    results = analysis['results']
    summary = results['summary']
    
    elements.append(Paragraph("Özet", heading_style))
    summary_data = [
        ["Toplam Hasar", str(summary['total_damages'])],
        ["Etkilenen Parça", str(summary['affected_parts'])],
        ["Ortalama Şiddet", f"{summary['average_severity']}/5"],
        ["Risk Seviyesi", summary['risk_level']]
    ]
    summary_table = Table(summary_data, colWidths=[6*cm, 6*cm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F5F5F7')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1D1D1F')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E5E7'))
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))
    
    # Damage details
    if results['damages']:
        elements.append(Paragraph("Hasar Detayları", heading_style))
        damage_data = [["Hasar Tipi", "Parça", "Güven", "Şiddet"]]
        for d in results['damages']:
            severity_dots = "●" * d['severity'] + "○" * (5 - d['severity'])
            damage_data.append([
                d['type_tr'],
                d['part_tr'] or "Belirsiz",
                f"%{d['confidence']}",
                severity_dots
            ])
        
        damage_table = Table(damage_data, colWidths=[4*cm, 4*cm, 2.5*cm, 3.5*cm])
        damage_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#000000')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E5E7')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#FAFAFA')])
        ]))
        elements.append(damage_table)
    else:
        elements.append(Paragraph("Hasar tespit edilmedi.", normal_style))
    
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("Bu rapor AutoDamageID yapay zeka sistemi tarafından otomatik olarak oluşturulmuştur.", 
                              ParagraphStyle('Footer', parent=normal_style, fontSize=9, textColor=colors.HexColor('#86868B'))))
    
    doc.build(elements)
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=hasar-raporu-{analysis_id[:8]}.pdf"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
