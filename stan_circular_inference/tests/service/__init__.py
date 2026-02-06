import pytest
import sys

def main():
    # Run tests with verbose output
    result = pytest.main([
        "tests/service/test.py",
        "-v",  # verbose output
        "--tb=short",  # shorter traceback
    ])
    
    sys.exit(result)

if __name__ == "__main__":
    main()