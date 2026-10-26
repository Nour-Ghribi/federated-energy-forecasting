"""CLI compatibility entrypoint; run after installing pulse-predict."""
import sys
from pulse_predict.cli import main

if __name__ == "__main__":
    main(['federated'] + sys.argv[1:])
