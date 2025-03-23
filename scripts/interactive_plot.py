import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QSlider, QLabel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

if not __name__ == "__main__":
    from . import anomaly_detection as ad


class AnomalyDetector(QWidget):
    def __init__(self,
                 irradiance_series,
                 location,
                 horizon=120,
                 tolerance=1,
                 night_tol=10,
                 outlier_tol = 1.2):
        super().__init__()
        self.series = irradiance_series
        self.horizon = horizon
        self.tolerance = tolerance
        self.night_tol = night_tol
        self.outlier_tol = outlier_tol
        self.location = location
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # Horizon slider
        self.horizon_label = QLabel(f"Linear fit horizon: {self.horizon}")
        layout.addWidget(self.horizon_label)

        self.horizon_slider = QSlider()
        self.horizon_slider.setOrientation(1)  # Vertical
        self.horizon_slider.setMinimum(3)
        self.horizon_slider.setMaximum(30)
        self.horizon_slider.setValue(int(self.horizon / 10))  # Remember to change scale in update_parameters() too
        self.horizon_slider.setTickInterval(3)
        self.horizon_slider.setTickPosition(QSlider.TicksBelow)
        self.horizon_slider.valueChanged.connect(self.update_parameters)
        layout.addWidget(self.horizon_slider)
       
        # Tolerance slider
        self.tolerance_label = QLabel(f"Linear fit tolerance: {self.tolerance:.2f}")
        layout.addWidget(self.tolerance_label)
        
        self.tolerance_slider = QSlider()
        self.tolerance_slider.setOrientation(1)  # Vertical
        self.tolerance_slider.setMinimum(-5)
        self.tolerance_slider.setMaximum(5)
        self.tolerance_slider.setValue(int(10 * np.log10(self.tolerance)))  # Scale up for precision
        self.tolerance_slider.setTickInterval(1)
        self.tolerance_slider.setTickPosition(QSlider.TicksBelow)
        self.tolerance_slider.valueChanged.connect(self.update_parameters)
        layout.addWidget(self.tolerance_slider)        
        
        # Night TOL slider
        self.night_label = QLabel(f"Lower threshold: {self.night_tol} [W/m^2]")
        layout.addWidget(self.night_label)
        
        self.night_slider = QSlider()
        self.night_slider.setOrientation(1)  # Vertical
        self.night_slider.setMinimum(0)
        self.night_slider.setMaximum(10)
        self.night_slider.setValue(self.night_tol)
        self.night_slider.setTickInterval(1)
        self.night_slider.setTickPosition(QSlider.TicksBelow)
        self.night_slider.valueChanged.connect(self.update_parameters)
        layout.addWidget(self.night_slider)      

        # Outlier tolerance:
        self.outlier_tol_label = QLabel(f"Outlier tolerance: {self.outlier_tol:.0%}")
        layout.addWidget(self.outlier_tol_label)

        self.outlier_tol_slider = QSlider()
        self.outlier_tol_slider.setOrientation(1)  # Vertical
        self.outlier_tol_slider.setMinimum(10)
        self.outlier_tol_slider.setMaximum(15)
        self.outlier_tol_slider.setValue(int(self.outlier_tol * 10))  # Remember to change scale in update_parameters() too
        self.outlier_tol_slider.setTickInterval(1)
        self.outlier_tol_slider.setTickPosition(QSlider.TicksBelow)
        self.outlier_tol_slider.valueChanged.connect(self.update_parameters)
        layout.addWidget(self.outlier_tol_slider)


        # Matplotlib Figure
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Initial plot
        self.update_plot()
        
        self.setLayout(layout)
    
    def update_parameters(self):
        self.horizon = 10 * self.horizon_slider.value()
        self.tolerance = 10 ** (self.tolerance_slider.value() / 10)
        self.night_tol = self.night_slider.value()
        self.outlier_tol = self.outlier_tol_slider.value() / 10

        self.horizon_label.setText(f"Linear fit horizon: {self.horizon}")
        self.tolerance_label.setText(f"Linear fit tolerance: {self.tolerance:.2f}")
        self.night_label.setText(f"Lower threshold: {self.night_tol} [W/m^2]")
        self.outlier_tol_label.setText(f"Outlier tolerance: {self.outlier_tol:.0%}")


        self.update_plot()  # Re-run detection and update plot
    
    def update_plot(self):
        anomaly_mask = ad.anomaly_linear(self.series,
                                         self.location,
                                         self.horizon,
                                         self.tolerance)
        
        anomaly_mask2, csky_threshold = ad.anomaly_clearsky(self.series,
                                                            self.location,
                                                            "ghi",
                                                            self.outlier_tol,
                                                            self.night_tol)

        self.ax.clear()
        self.ax.plot(self.series.index,
                     self.series,
                     '.',
                     markersize=1.5,
                     label="Irradiance",
                     color="blue")
        
        self.ax.scatter(self.series.index[anomaly_mask2],
                        self.series.loc[anomaly_mask2],
                        label="Clearsky outliers",
                        color="green")

        self.ax.plot(self.series.index,
                     csky_threshold,
                     linestyle="dashed",                     
                     label="Clearsky outlier threshold",
                     color="black",
                     linewidth=0.5)
        
        self.ax.scatter(self.series.index[anomaly_mask],
                        self.series.loc[anomaly_mask],
                        label="Linear anomaly (daytime)",
                        color="red")
        
        self.ax.scatter(self.series.index[anomaly_mask & anomaly_mask2],
                        self.series.loc[anomaly_mask & anomaly_mask2],
                        label="Both",
                        color="purple")
        
        self.ax.legend()
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Irradiance")
        self.ax.set_title("Anomaly detection example")
        
        self.canvas.draw()

# Example code for debug
if __name__ == "__main__":
    import synthetic_data_generation as sdg
    import anomaly_detection as ad
    ghi = sdg.SyntheticIrradiance()
    ghi.add_sensor_disconnect()
    ghi.add_noise()
    ghi.add_outliers()

    # Run the application
    app = QApplication(sys.argv)
    window = AnomalyDetector(ghi.series, ghi.location)
    window.show()
    sys.exit(app.exec_())