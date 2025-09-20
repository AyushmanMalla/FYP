# File: env_check.py
# A simple script to diagnose the Python environment on NSCC.

import sys

print("--- 🐍 Python Environment Check ---")
print(f"Python version: {sys.version}")

try:
    import pandas as pd
    print(f"✅ pandas version: {pd.__version__}")
except ImportError as e:
    print(f"❌ ERROR: Failed to import pandas. {e}")

try:
    import sklearn
    print(f"✅ scikit-learn version: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ ERROR: Failed to import scikit-learn. {e}")

try:
    from PIL import Image
    print(f"✅ Pillow (PIL) is installed.")
except ImportError as e:
    print(f"❌ ERROR: Failed to import Pillow (PIL). {e}")

try:
    import torch
    print(f"✅ torch version: {torch.__version__}")
    
    # --- GPU Check ---
    is_cuda = torch.cuda.is_available()
    print(f"   CUDA available: {is_cuda}")
    if is_cuda:
        print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print("   WARNING: No GPU detected by PyTorch.")

except ImportError as e:
    print(f"❌ ERROR: Failed to import torch. {e}")


print("\n--- ✅ Environment check finished! ---")
