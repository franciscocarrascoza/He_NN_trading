// FIX: ControlWidget implementation for training control (no trading UI) per spec

#include "controlwidget.h"  // FIX: control widget header
#include <QVBoxLayout>  // FIX: vertical layout
#include <QHBoxLayout>  // FIX: horizontal layout
#include <QFormLayout>  // FIX: form layout for label/widget pairs
#include <QGroupBox>  // FIX: group box for organizing controls

ControlWidget::ControlWidget(QWidget *parent)  // FIX: constructor implementation
    : QWidget(parent)  // FIX: call base constructor
{
    setupUi();  // FIX: initialize UI components
}

ControlWidget::~ControlWidget()  // FIX: destructor implementation
{
    // FIX: Qt will auto-delete child objects
}

void ControlWidget::setupUi()  // FIX: UI initialization implementation
{
    // FIX: Create group box for training controls
    QGroupBox *controlGroup = new QGroupBox("Training Control", this);  // FIX: create group box
    QVBoxLayout *groupLayout = new QVBoxLayout(controlGroup);  // FIX: create vertical layout

    // FIX: Create form layout for parameter controls
    QFormLayout *formLayout = new QFormLayout();  // FIX: create form layout

    // FIX: Timeframe selector per spec
    timeframeCombo = new QComboBox();  // FIX: create combo box
    timeframeCombo->addItems({"1m", "5m", "15m", "30m", "1h", "4h", "1d"});  // FIX: add timeframe options
    timeframeCombo->setCurrentText("1h");  // FIX: default to 1h per spec
    formLayout->addRow("Timeframe:", timeframeCombo);  // FIX: add to form

    // FIX: Forecast horizon spinner per spec
    horizonSpin = new QSpinBox();  // FIX: create spin box
    horizonSpin->setRange(1, 100);  // FIX: set range
    horizonSpin->setValue(1);  // FIX: default to horizon=1 per spec
    formLayout->addRow("Forecast Horizon:", horizonSpin);  // FIX: add to form

    // FIX: Calibration method selector per spec
    calibrationMethodCombo = new QComboBox();  // FIX: create combo box
    calibrationMethodCombo->addItems({"abs", "std_gauss"});  // FIX: add calibration methods per spec
    calibrationMethodCombo->setCurrentText("std_gauss");  // FIX: default to std_gauss per spec
    formLayout->addRow("Calibration Method:", calibrationMethodCombo);  // FIX: add to form

    // FIX: p_up source selector per spec
    pUpSourceCombo = new QComboBox();  // FIX: create combo box
    pUpSourceCombo->addItems({"cdf", "logit"});  // FIX: add p_up source options per spec
    pUpSourceCombo->setCurrentText("cdf");  // FIX: default to cdf per spec
    formLayout->addRow("p_up Source:", pUpSourceCombo);  // FIX: add to form

    // FIX: CV enable checkbox per spec
    cvCheckbox = new QCheckBox("Enable Cross-Validation");  // FIX: create checkbox
    cvCheckbox->setChecked(true);  // FIX: default to enabled per spec
    formLayout->addRow(cvCheckbox);  // FIX: add to form

    groupLayout->addLayout(formLayout);  // FIX: add form to group layout

    // FIX: Create button layout
    QHBoxLayout *buttonLayout = new QHBoxLayout();  // FIX: create horizontal layout

    // FIX: Start button per spec
    startButton = new QPushButton("Start Training");  // FIX: create start button
    startButton->setStyleSheet("background-color: #00aa00; color: white;");  // FIX: green button
    connect(startButton, &QPushButton::clicked, this, &ControlWidget::onStartButtonClicked);  // FIX: connect signal
    buttonLayout->addWidget(startButton);  // FIX: add to layout

    // FIX: Stop button per spec
    stopButton = new QPushButton("Stop Training");  // FIX: create stop button
    stopButton->setStyleSheet("background-color: #aa0000; color: white;");  // FIX: red button
    stopButton->setEnabled(false);  // FIX: disabled by default
    connect(stopButton, &QPushButton::clicked, this, &ControlWidget::onStopButtonClicked);  // FIX: connect signal
    buttonLayout->addWidget(stopButton);  // FIX: add to layout

    groupLayout->addLayout(buttonLayout);  // FIX: add button layout to group

    // FIX: Set main layout
    QVBoxLayout *mainLayout = new QVBoxLayout(this);  // FIX: create vertical layout
    mainLayout->addWidget(controlGroup);  // FIX: add control group
    mainLayout->addStretch();  // FIX: add stretch to push controls to top
}

void ControlWidget::onStartButtonClicked()  // FIX: start button handler
{
    // FIX: Emit training start signal
    emit trainingStartRequested();  // FIX: emit signal

    // FIX: Update button states
    startButton->setEnabled(false);  // FIX: disable start button
    stopButton->setEnabled(true);  // FIX: enable stop button
}

void ControlWidget::onStopButtonClicked()  // FIX: stop button handler
{
    // FIX: Emit training stop signal
    emit trainingStopRequested();  // FIX: emit signal

    // FIX: Update button states
    startButton->setEnabled(true);  // FIX: enable start button
    stopButton->setEnabled(false);  // FIX: disable stop button
}
