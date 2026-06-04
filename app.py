import os, uuid, zipfile, shutil, logging, traceback, threading, time, json, queue
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from skimage.filters import threshold_sauvola
import base64
from io import BytesIO
from PIL import Image

logging.basicConfig(level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('binarocr')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024 * 1024  # 20 Go
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

ALLOWED_EXTENSIONS = {'png','jpg','jpeg','tiff','tif','bmp','webp','zip'}
IMG_EXTENSIONS     = {'.png','.jpg','.jpeg','.tiff','.tif','.bmp','.webp'}
LARGE_FILE_BYTES   = 5 * 1024 * 1024
MAX_TIFF_KB        = 300
DPI_FALLBACK_STEPS = [300, 250, 200, 150]

# ── Job store ──────────────────────────────────────────────────────────────────
# jobs[session_id] = { status, total_images, done_images, folders: {name: {...}}, ... }
jobs = {}
job_queues = defaultdict(list)   # session_id → list of SSE subscriber queues
jobs_lock  = threading.Lock()

def job_update(sid, patch):
    with jobs_lock:
        jobs[sid].update(patch)
    event = json.dumps(jobs[sid], default=str)
    for q in job_queues[sid]:
        q.put(event)

def job_folder_update(sid, folder, patch):
    with jobs_lock:
        jobs[sid]['folders'][folder].update(patch)
    event = json.dumps(jobs[sid], default=str)
    for q in job_queues[sid]:
        q.put(event)

# ── Helpers ────────────────────────────────────────────────────────────────────
def allowed_file(f): return '.' in f and f.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(413)
def too_large(e):
    msg = f"Fichier trop volumineux — limite : {app.config['MAX_CONTENT_LENGTH']//1024//1024//1024} Go"
    log.warning('413 — %s', msg)
    return jsonify({'error': msg}), 413

def load_gray_from_bytes(img_bytes):
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None: raise ValueError('cv2.imdecode a retourné None')
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def get_dpi_from_bytes(img_bytes):
    try:
        pil = Image.open(BytesIO(img_bytes))
        info = pil.info.get('dpi', (72,72))
        return int(round(info[0])) if info[0] > 0 else 72
    except: return 72

def img_to_base64_preview(arr, max_dim=800):
    h,w = arr.shape[:2]
    if max(h,w) > max_dim:
        s = max_dim/max(h,w)
        arr = cv2.resize(arr,(int(w*s),int(h*s)),interpolation=cv2.INTER_AREA)
    mode = 'L' if len(arr.shape)==2 else 'RGB'
    buf = BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def save_tiff_ccitt4(arr, path, dpi=300):
    Image.fromarray(arr.astype(np.uint8)).convert('1').save(
        path, format='TIFF', compression='group4', dpi=(dpi,dpi))
    kb = os.path.getsize(path)/1024
    log.info('TIFF : %s  %.1fKo  CCITT4  %dDPI', path, kb, dpi)
    return kb

def save_tiff_with_limit(arr, path):
    for dpi in DPI_FALLBACK_STEPS:
        kb = save_tiff_ccitt4(arr, path, dpi)
        if kb <= MAX_TIFF_KB:
            if dpi < 300: log.warning('DPI abaissé → %d (%.1fKo)', dpi, kb)
            return dpi, kb
    return DPI_FALLBACK_STEPS[-1], kb

def ensure_300dpi(gray, src_dpi):
    if src_dpi >= 300: return gray
    s = 300/src_dpi
    return cv2.resize(gray,(int(gray.shape[1]*s),int(gray.shape[0]*s)),interpolation=cv2.INTER_LANCZOS4)

def apply_clahe(gray, clip=2.0): return cv2.createCLAHE(clipLimit=clip,tileGridSize=(8,8)).apply(gray)

def deskew(gray):
    t = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    c = np.column_stack(np.where(t>0))
    if len(c)<10: return gray
    a = cv2.minAreaRect(c)[-1]
    a = -(90+a) if a<-45 else -a
    if abs(a)<0.5: return gray
    h,w = gray.shape
    M = cv2.getRotationMatrix2D((w//2,h//2),a,1.0)
    return cv2.warpAffine(gray,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

def binarize(gray, p):
    if p.get('clahe',True): gray = apply_clahe(gray, float(p.get('clahe_clip',2.0)))
    if p.get('denoise',True):
        k = int(p.get('denoise_ksize',3)); k = k if k%2==1 else k+1
        gray = cv2.medianBlur(gray,k) if p.get('denoise_type')=='median' else cv2.GaussianBlur(gray,(k,k),0)
    if p.get('deskew',True): gray = deskew(gray)
    m = p.get('method','sauvola')
    if m=='sauvola':
        win=int(p.get('sauvola_window',25)); win=win if win%2==1 else win+1
        thresh=threshold_sauvola(gray,window_size=win,k=float(p.get('sauvola_k',0.3)))
        binary=(gray>thresh).astype(np.uint8)*255
    elif m=='otsu':
        _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif m=='adaptive_mean':
        bl=int(p.get('adaptive_block',31)); bl=bl if bl%2==1 else bl+1
        binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,bl,int(p.get('adaptive_c',10)))
    elif m=='adaptive_gaussian':
        bl=int(p.get('adaptive_block',31)); bl=bl if bl%2==1 else bl+1
        binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,bl,int(p.get('adaptive_c',10)))
    else:
        _,binary=cv2.threshold(gray,int(p.get('manual_thresh',128)),255,cv2.THRESH_BINARY)
    ks=int(p.get('morph_ksize',2))
    if p.get('morph',True) and ks>0:
        K=np.ones((ks,ks),np.uint8); op=p.get('morph_op','opening')
        if op=='opening': binary=cv2.morphologyEx(binary,cv2.MORPH_OPEN,K)
        elif op=='closing': binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,K)
        elif op=='both':
            binary=cv2.morphologyEx(binary,cv2.MORPH_OPEN,K)
            binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,K)
    return binary

def parse_bool_params(d):
    p=dict(d)
    for k in ['denoise','clahe','deskew','morph']: p[k]=p.get(k,'false').lower()=='true'
    return p

PRESETS = {
    'fast':     dict(method='otsu',denoise=True,denoise_type='gaussian',denoise_ksize=3,clahe=False,deskew=False,morph=False,morph_op='opening',morph_ksize=1),
    'balanced': dict(method='sauvola',denoise=True,denoise_type='gaussian',denoise_ksize=3,clahe=True,clahe_clip=2.0,deskew=True,morph=True,morph_op='opening',morph_ksize=2,sauvola_window=25,sauvola_k=0.3),
    'quality':  dict(method='sauvola',denoise=True,denoise_type='gaussian',denoise_ksize=5,clahe=True,clahe_clip=3.0,deskew=True,morph=True,morph_op='both',morph_ksize=2,sauvola_window=35,sauvola_k=0.35),
}

# ── Batch worker ───────────────────────────────────────────────────────────────
def batch_worker(sid, zip_path, params, out_root):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Regrouper les membres par dossier
            folder_map = defaultdict(list)
            for m in zf.infolist():
                if m.file_size == 0: continue
                if m.filename.startswith('__MACOSX'): continue
                ext = os.path.splitext(m.filename)[1].lower()
                if ext not in IMG_EXTENSIONS: continue
                folder = os.path.dirname(m.filename) or '__root__'
                folder_map[folder].append(m)

            total_images = sum(len(v) for v in folder_map.values())
            folders_info = {}
            for fname, members in folder_map.items():
                display = fname if fname != '__root__' else '(racine)'
                folders_info[fname] = {
                    'display': display,
                    'total': len(members),
                    'done': 0,
                    'errors': 0,
                    'status': 'pending',
                    'zip_file': None,
                }

            job_update(sid, {
                'status': 'running',
                'total_images': total_images,
                'done_images': 0,
                'error_images': 0,
                'total_folders': len(folder_map),
                'done_folders': 0,
                'folders': folders_info,
            })

            done_total = 0

            for folder_name, members in folder_map.items():
                folder_out = os.path.join(out_root, folder_name.replace('/', '_').replace('\\','_'))
                os.makedirs(folder_out, exist_ok=True)
                job_folder_update(sid, folder_name, {'status': 'running'})

                for idx, member in enumerate(members, 1):
                    fname = os.path.basename(member.filename)
                    try:
                        img_bytes = zf.read(member.filename)
                        file_size = len(img_bytes)
                        is_large  = file_size > LARGE_FILE_BYTES

                        src_dpi = get_dpi_from_bytes(img_bytes)
                        gray    = load_gray_from_bytes(img_bytes)
                        del img_bytes

                        gray   = ensure_300dpi(gray, src_dpi)
                        binary = binarize(gray, params)

                        out_name = os.path.splitext(fname)[0] + '_bin.tiff'
                        out_path = os.path.join(folder_out, out_name)
                        dpi_used, size_kb = save_tiff_with_limit(binary, out_path)

                        log.info('[%s][%d/%d] OK %s  %dx%d  %dDPI  %.1fKo',
                                 folder_name, idx, len(members), fname,
                                 gray.shape[1], gray.shape[0], dpi_used, size_kb)

                        img_result = {
                            'filename': fname, 'output': out_name, 'status': 'ok',
                            'src_dpi': src_dpi, 'out_dpi': dpi_used,
                            'size_kb': round(size_kb,1),
                        }
                        if not is_large:
                            img_result['binarized'] = img_to_base64_preview(binary)
                        del gray, binary

                        done_total += 1
                        with jobs_lock:
                            jobs[sid]['done_images'] = done_total
                            jobs[sid]['folders'][folder_name]['done'] += 1
                            jobs[sid]['folders'][folder_name].setdefault('images', []).append(img_result)

                    except Exception as e:
                        log.error('[%s] FAIL %s → %s', folder_name, fname, e)
                        done_total += 1
                        with jobs_lock:
                            jobs[sid]['done_images'] = done_total
                            jobs[sid]['error_images'] = jobs[sid].get('error_images',0)+1
                            jobs[sid]['folders'][folder_name]['errors'] += 1
                            jobs[sid]['folders'][folder_name]['done'] += 1
                            jobs[sid]['folders'][folder_name].setdefault('images', []).append(
                                {'filename': fname, 'status': 'error', 'error': str(e)})

                    # Push SSE update après chaque image
                    event = json.dumps(jobs[sid], default=str)
                    for q in job_queues[sid]: q.put(event)

                # ── Dossier terminé : créer son ZIP ──
                folder_zip_name = f"{sid}_{folder_name.replace('/','_').replace('\\','_') or 'root'}.zip"
                folder_zip_path = os.path.join(app.config['OUTPUT_FOLDER'], folder_zip_name)
                with zipfile.ZipFile(folder_zip_path, 'w', zipfile.ZIP_DEFLATED) as zout:
                    for f in os.listdir(folder_out):
                        zout.write(os.path.join(folder_out, f), f)
                folder_zip_kb = os.path.getsize(folder_zip_path)/1024
                log.info('Dossier "%s" terminé → %s (%.1fKo)', folder_name, folder_zip_name, folder_zip_kb)

                with jobs_lock:
                    jobs[sid]['folders'][folder_name].update({
                        'status': 'done',
                        'zip_file': folder_zip_name,
                        'zip_kb': round(folder_zip_kb, 1),
                    })
                    jobs[sid]['done_folders'] = jobs[sid].get('done_folders',0)+1

                # Push SSE — le frontend va déclencher le téléchargement auto
                event = json.dumps(jobs[sid], default=str)
                for q in job_queues[sid]: q.put(event)

                shutil.rmtree(folder_out, ignore_errors=True)

        job_update(sid, {'status': 'done'})
        log.info('Job [%s] terminé : %d/%d images', sid, jobs[sid]['done_images'], total_images)

    except Exception as e:
        log.error('Job [%s] erreur fatale : %s\n%s', sid, e, traceback.format_exc())
        job_update(sid, {'status': 'error', 'error': str(e)})
    finally:
        shutil.rmtree(os.path.dirname(zip_path), ignore_errors=True)
        # Fermer les queues abonnées
        for q in job_queues.get(sid, []):
            q.put(None)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/presets')
def get_presets(): return jsonify(PRESETS)

@app.route('/api/process', methods=['POST'])
def process_image():
    log.debug('=== /api/process ===')
    if 'image' not in request.files:
        return jsonify({'error': f"Champ 'image' absent. Reçu : {list(request.files.keys())}"}), 400
    file = request.files['image']
    if file.filename == '': return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    if not allowed_file(file.filename):
        ext = file.filename.rsplit('.',1)[-1] if '.' in file.filename else '?'
        return jsonify({'error': f"Extension '.{ext}' non supportée"}), 400

    params = parse_bool_params(request.form.to_dict())
    sid    = str(uuid.uuid4())[:8]
    fname  = secure_filename(file.filename)
    upath  = os.path.join(app.config['UPLOAD_FOLDER'], f"{sid}_{fname}")
    file.save(upath)

    try:
        img_bytes = open(upath,'rb').read()
        file_size = len(img_bytes)
        is_large  = file_size > LARGE_FILE_BYTES
        src_dpi   = get_dpi_from_bytes(img_bytes)
        gray      = load_gray_from_bytes(img_bytes); del img_bytes
        log.info('Image : %s  %s  %dDPI  %.1fMo', fname, gray.shape, src_dpi, file_size/1024/1024)

        gray     = ensure_300dpi(gray, src_dpi)
        binary   = binarize(gray, params)
        out_name = f"{sid}_binarized_{fname.rsplit('.',1)[0]}.tiff"
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_name)
        dpi_used, size_kb = save_tiff_with_limit(binary, out_path)

        return jsonify({
            'output_file': out_name, 'session_id': sid,
            'src_dpi': src_dpi, 'out_dpi': dpi_used,
            'size_kb': round(size_kb,1), 'large_file': is_large,
            'shape': list(gray.shape),
            'original':  None if is_large else img_to_base64_preview(gray),
            'binarized': None if is_large else img_to_base64_preview(binary),
        })
    except Exception as e:
        log.error('%s\n%s', e, traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(upath): os.remove(upath)

@app.route('/api/process-batch', methods=['POST'])
def process_batch():
    log.debug('=== /api/process-batch ===')
    if 'archive' not in request.files:
        return jsonify({'error': f"Champ 'archive' absent. Reçu : {list(request.files.keys())}"}), 400
    file = request.files['archive']
    if not file.filename.lower().endswith('.zip'):
        return jsonify({'error': 'Seuls les fichiers .zip sont acceptés'}), 400

    params = parse_bool_params(request.form.to_dict())
    sid    = str(uuid.uuid4())[:8]
    zip_dir = os.path.join(app.config['UPLOAD_FOLDER'], sid)
    out_root = os.path.join(app.config['OUTPUT_FOLDER'], sid)
    os.makedirs(zip_dir, exist_ok=True)
    os.makedirs(out_root, exist_ok=True)

    zip_path = os.path.join(zip_dir, 'input.zip')
    log.info('Batch [%s] réception ZIP…', sid)
    bytes_written = 0
    chunk_size = 4*1024*1024
    with open(zip_path,'wb') as f:
        while True:
            chunk = file.stream.read(chunk_size)
            if not chunk: break
            f.write(chunk); bytes_written+=len(chunk)
    log.info('Batch [%s] ZIP reçu %.1fMo', sid, bytes_written/1024/1024)

    # Initialiser le job
    with jobs_lock:
        jobs[sid] = {
            'session_id': sid, 'status': 'starting',
            'total_images': 0, 'done_images': 0, 'error_images': 0,
            'total_folders': 0, 'done_folders': 0,
            'folders': {},
        }
        job_queues[sid] = []

    # Lancer le worker en arrière-plan
    t = threading.Thread(target=batch_worker, args=(sid, zip_path, params, out_root), daemon=True)
    t.start()

    return jsonify({'session_id': sid, 'status': 'started'})

@app.route('/api/job/<sid>')
def job_status(sid):
    with jobs_lock:
        job = jobs.get(sid)
    if not job: return jsonify({'error': 'Job introuvable'}), 404
    return jsonify(job)

@app.route('/api/job/<sid>/stream')
def job_stream(sid):
    """SSE endpoint — le frontend s'y abonne pour recevoir les mises à jour en temps réel."""
    q = queue.Queue()
    with jobs_lock:
        if sid not in job_queues: job_queues[sid] = []
        job_queues[sid].append(q)

    def generate():
        # Envoyer l'état actuel immédiatement
        with jobs_lock:
            current = jobs.get(sid, {})
        yield f"data: {json.dumps(current, default=str)}\n\n"
        try:
            while True:
                try:
                    event = q.get(timeout=30)
                    if event is None: break
                    yield f"data: {event}\n\n"
                except queue.Empty:
                    yield ": heartbeat\n\n"
        finally:
            with jobs_lock:
                try: job_queues[sid].remove(q)
                except ValueError: pass

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

@app.route('/api/download/<path:filename>')
def download_file(filename):
    safe = os.path.basename(secure_filename(filename))
    path = os.path.join(app.config['OUTPUT_FOLDER'], safe)
    if not os.path.exists(path):
        return jsonify({'error': 'Fichier introuvable'}), 404
    ext = safe.rsplit('.',1)[-1].lower() if '.' in safe else ''
    mime = {'zip':'application/zip','tiff':'image/tiff','tif':'image/tiff','png':'image/png'}.get(ext,'application/octet-stream')
    return send_file(path, as_attachment=True, mimetype=mime)

@app.route('/api/debug')
def debug_info():
    import skimage, PIL
    with jobs_lock:
        active = {k:v['status'] for k,v in jobs.items()}
    return jsonify({
        'status':'ok','active_jobs': active,
        'config':{'max_content_gb': app.config['MAX_CONTENT_LENGTH']//1024**3,
                  'max_tiff_kb': MAX_TIFF_KB, 'dpi_fallback': DPI_FALLBACK_STEPS},
        'versions':{'opencv':cv2.__version__,'scikit-image':skimage.__version__,
                    'pillow':PIL.__version__,'numpy':np.__version__},
    })

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    app.run(debug=True, port=5000, threaded=True)
