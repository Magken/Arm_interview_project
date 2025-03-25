import os
import sys

# Ensure Windows can locate required DLLs (adjust paths as needed)
os.add_dll_directory("E:\\Intel\\oneAPI\\mkl\\latest\\bin")
os.add_dll_directory("E:\\Coding Stuff\\Arm_interview_project\\build")
os.add_dll_directory("C:\\Users\\Magjun\\AppData\\Local\\Programs\\Python\\Python311")

# Add build directory to Python module search path
sys.path.append("E:\\Coding Stuff\\Arm_interview_project\\build")