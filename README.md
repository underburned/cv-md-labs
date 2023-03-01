# ОИМУ ЛР

## Лабораторная работа 1

1. Выбрать любой алгоритм (или несколько) обработки изображений:
   - переход из одного цветового пространства в другое
   - гамма-коррекция
   - контрастирование
   - изменение баланса белого
   - пороговая обработка
   - ...  

2. Получить доступ к камере ноутбука/мобильного устройства или подготовить набор изображений со смартфона
3. Применить выбранный алгоритм (алгоритмы) к изображениям, привести результаты обработки.

Полезные ссылки:
- [OpenCV Image Filtering](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html)
- [OpenCV Color Space Conversions](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html) - 
основной метод `cvtColor` и различные поля `enum cv::ColorConversionCodes`.
- [OpenCV Color conversions](https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html)
- [Туторы на питоне](https://docs.opencv.org/4.7.0/d6/d00/tutorial_py_root.html)
- [OpenCV Getting Started with Videos, Python](https://docs.opencv.org/4.7.0/dd/d43/tutorial_py_video_display.html)


- [Пример jupyter-notebook'а](examples/l1_gamma.ipynb)
- [Получение доступа к камере на Android](examples/AndroidCameraX.md)