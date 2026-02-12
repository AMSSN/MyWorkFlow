import pyautogui

img = r"E:\CodePython\MyWorkFlow\PythonProject\AutoClick\resource\shouhuo.png"


location = pyautogui.locateCenterOnScreen(img, confidence=0.7)
if location is not None:
    pyautogui.moveTo(location.x, location.y, duration=0.2)

