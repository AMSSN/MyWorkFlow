打包命令：

pyinstaller --onefile -F -w -i icons/inspect.png --add-data "3rdparty/turbojpeg/libturbojpeg.dll;." MainWindow.py

pyinstaller --onedir -w -i icons/inspect.png MainWindow.py

nuitka --standalone --onefile --windows-disable-console --mingw64 --nofollow-imports --show-memory --show-progress --plugin-enable=pyqt5 --include-qt-plugins=sensible,styles --windows-icon-from-ico=icons/inspect.png --follow-import-to=ui --output-dir=output MainWindow.py
