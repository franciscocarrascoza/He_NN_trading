// FIX: ControlWidget header for training control panel (no trading UI) per spec

#ifndef CONTROLWIDGET_H  // FIX: header guard
#define CONTROLWIDGET_H

#include <QWidget>  // FIX: Qt widget base class
#include <QPushButton>  // FIX: push button
#include <QComboBox>  // FIX: combo box for selections
#include <QSpinBox>  // FIX: spin box for numeric inputs
#include <QCheckBox>  // FIX: checkbox for toggles

class ControlWidget : public QWidget  // FIX: ControlWidget class per spec
{
    Q_OBJECT  // FIX: Qt meta-object macro

public:
    explicit ControlWidget(QWidget *parent = nullptr);  // FIX: constructor
    ~ControlWidget();  // FIX: destructor

signals:
    void trainingStartRequested();  // FIX: training start signal
    void trainingStopRequested();  // FIX: training stop signal

private slots:
    void onStartButtonClicked();  // FIX: start button handler
    void onStopButtonClicked();  // FIX: stop button handler

private:
    void setupUi();  // FIX: initialize UI components

    // FIX: Control widgets per spec (no trading controls!)
    QPushButton *startButton;  // FIX: start training button
    QPushButton *stopButton;  // FIX: stop training button
    QComboBox *timeframeCombo;  // FIX: timeframe selector per spec
    QSpinBox *horizonSpin;  // FIX: forecast horizon spinner per spec
    QComboBox *calibrationMethodCombo;  // FIX: calibration method selector per spec
    QComboBox *pUpSourceCombo;  // FIX: p_up source selector per spec
    QCheckBox *cvCheckbox;  // FIX: CV enable checkbox per spec
};

#endif  // FIX: close header guard
