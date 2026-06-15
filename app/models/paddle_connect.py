"""
Singleton to connect to PaddleOCR instance.
"""
import logging
log = logging.getLogger(__name__)

_ocr_instance = None
def _get_ocr():
    """
    Initialise PaddleOCR once and reuse across calls.
    CPU-only: use_gpu=False, use_angle_cls=True (handles rotated text).
    """
    global _ocr_instance
    if _ocr_instance is None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise ImportError(
                "PaddleOCR is not installed. Run:\n"
                "  pip install paddlepaddle paddleocr"
            ) from e

        log.info("Initialising PaddleOCR (CPU) — first call downloads model weights (~30 MB).")
        _ocr_instance = PaddleOCR(
            use_textline_orientation=True,   # Correct 90°/180°/270° text rotation
            lang="en",            # Change to "ch", "hi", etc. for other languages
            # use_gpu=False,        # CPU inference
            # show_log=False,       # Suppress PaddlePaddle verbose output
            enable_mkldnn=False,   # Intel MKL-DNN acceleration on x86 CPUs (free perf boost)
        )
        log.info("PaddleOCR ready.")
    return _ocr_instance

