#!/usr/bin/env python3
"""
process_folder.py — Traitement en ligne de commande d'un dossier d'images.

Usage :
    python process_folder.py <dossier_source> [dossier_sortie] [options]

Exemples :
    python process_folder.py ./scans
    python process_folder.py ./scans ./resultats --preset quality
    python process_folder.py ./scans --method otsu --no-deskew --no-morph
"""

import argparse
import logging
import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from skimage.filters import threshold_sauvola

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('process_folder')

IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp'}

PRESETS = {
    'fast': dict(method='otsu', denoise=True, denoise_type='gaussian', denoise_ksize=3,
                 clahe=False, deskew=False, morph=False, morph_op='opening', morph_ksize=1),
    'balanced': dict(method='sauvola', denoise=True, denoise_type='gaussian', denoise_ksize=3,
                     clahe=True, clahe_clip=2.0, deskew=True, morph=True,
                     morph_op='opening', morph_ksize=2, sauvola_window=25, sauvola_k=0.3),
    'quality': dict(method='sauvola', denoise=True, denoise_type='gaussian', denoise_ksize=5,
                    clahe=True, clahe_clip=3.0, deskew=True, morph=True,
                    morph_op='both', morph_ksize=2, sauvola_window=35, sauvola_k=0.35),
}


def apply_clahe(gray, clip=2.0, tile=8):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def deskew(gray):
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def binarize(gray, p):
    if p.get('clahe'):
        gray = apply_clahe(gray, clip=float(p.get('clahe_clip', 2.0)))

    if p.get('denoise'):
        ksize = int(p.get('denoise_ksize', 3))
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        if p.get('denoise_type') == 'median':
            gray = cv2.medianBlur(gray, ksize)
        else:
            gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    if p.get('deskew'):
        gray = deskew(gray)

    method = p.get('method', 'sauvola')
    if method == 'sauvola':
        win = int(p.get('sauvola_window', 25))
        win = win if win % 2 == 1 else win + 1
        k = float(p.get('sauvola_k', 0.3))
        thresh = threshold_sauvola(gray, window_size=win, k=k)
        binary = (gray > thresh).astype(np.uint8) * 255
    elif method == 'otsu':
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive_gaussian':
        block = int(p.get('adaptive_block', 31))
        block = block if block % 2 == 1 else block + 1
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, block, int(p.get('adaptive_c', 10)))
    else:
        _, binary = cv2.threshold(gray, int(p.get('manual_thresh', 128)), 255, cv2.THRESH_BINARY)

    if p.get('morph') and int(p.get('morph_ksize', 2)) > 0:
        kernel = np.ones((int(p['morph_ksize']), int(p['morph_ksize'])), np.uint8)
        op = p.get('morph_op', 'opening')
        if op == 'opening':
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        elif op == 'closing':
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif op == 'both':
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


def process_folder(src: Path, dst: Path, params: dict, recursive: bool = True):
    pattern = '**/*' if recursive else '*'
    images = [p for p in src.glob(pattern) if p.suffix.lower() in IMG_EXTENSIONS]

    if not images:
        log.warning('Aucune image trouvée dans %s', src)
        return

    dst.mkdir(parents=True, exist_ok=True)
    log.info('=' * 60)
    log.info('Source      : %s', src.resolve())
    log.info('Destination : %s', dst.resolve())
    log.info('Images      : %d fichier(s)', len(images))
    log.info('Méthode     : %s', params.get('method', 'sauvola'))
    log.info('=' * 60)

    ok_count = 0
    fail_count = 0
    t0 = time.time()

    for idx, img_path in enumerate(sorted(images), 1):
        t_img = time.time()
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError('cv2.imread a retourné None — fichier illisible ou corrompu')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            binary = binarize(gray, params)

            # Conserver la structure de sous-dossiers si récursif
            rel = img_path.relative_to(src)
            out_path = dst / rel.with_suffix('.png')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), binary)

            elapsed_ms = (time.time() - t_img) * 1000
            size_kb = img_path.stat().st_size / 1024
            log.info('[%d/%d] OK   %-40s  %dx%d  %.1f KB  %.0fms  -> %s',
                     idx, len(images),
                     img_path.name, gray.shape[1], gray.shape[0],
                     size_kb, elapsed_ms,
                     out_path.relative_to(dst.parent))
            ok_count += 1

        except Exception as e:
            log.error('[%d/%d] FAIL %-40s  -> %s', idx, len(images), img_path.name, e)
            fail_count += 1

    total = time.time() - t0
    log.info('=' * 60)
    log.info('Terminé en %.1fs  |  %d OK  |  %d erreur(s)', total, ok_count, fail_count)
    log.info('Résultats dans : %s', dst.resolve())
    log.info('=' * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Binarisation d\'un dossier d\'images pour OCR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('source', help='Dossier source contenant les images')
    parser.add_argument('output', nargs='?', help='Dossier de sortie (défaut: source_binarized)')
    parser.add_argument('--preset', choices=['fast', 'balanced', 'quality'],
                        default='balanced', help='Préset de paramètres (défaut: balanced)')
    parser.add_argument('--method', choices=['sauvola', 'otsu', 'adaptive_gaussian', 'manual'],
                        help='Surcharger la méthode de binarisation')
    parser.add_argument('--no-clahe', action='store_true', help='Désactiver CLAHE')
    parser.add_argument('--no-denoise', action='store_true', help='Désactiver le débruitage')
    parser.add_argument('--no-deskew', action='store_true', help='Désactiver le deskew')
    parser.add_argument('--no-morph', action='store_true', help='Désactiver la morphologie')
    parser.add_argument('--sauvola-window', type=int, help='Taille fenêtre Sauvola (défaut: 25)')
    parser.add_argument('--sauvola-k', type=float, help='Facteur k Sauvola (défaut: 0.3)')
    parser.add_argument('--no-recursive', action='store_true',
                        help='Ne pas parcourir les sous-dossiers')

    args = parser.parse_args()

    src = Path(args.source)
    if not src.is_dir():
        log.error('Le dossier source n\'existe pas : %s', src)
        sys.exit(1)

    dst = Path(args.output) if args.output else src.parent / (src.name + '_binarized')

    # Partir du préset puis appliquer les surcharges
    params = dict(PRESETS[args.preset])

    if args.method:
        params['method'] = args.method
    if args.no_clahe:
        params['clahe'] = False
    if args.no_denoise:
        params['denoise'] = False
    if args.no_deskew:
        params['deskew'] = False
    if args.no_morph:
        params['morph'] = False
    if args.sauvola_window:
        params['sauvola_window'] = args.sauvola_window
    if args.sauvola_k:
        params['sauvola_k'] = args.sauvola_k

    process_folder(src, dst, params, recursive=not args.no_recursive)


if __name__ == '__main__':
    main()
