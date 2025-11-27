// FIX: ControlWidget implementation for training control (no trading UI) per spec

#include "controlwidget.h"  // FIX: control widget header
#include <QVBoxLayout>  // FIX: vertical layout
#include <QHBoxLayout>  // FIX: horizontal layout
#include <QFormLayout>  // FIX: form layout for label/widget pairs
#include <QGroupBox>  // FIX: group box for organizing controls
#include <QScrollArea>  // FIX: scroll area for long parameter list
#include <QJsonObject>  // FIX: JSON object for parameter collection

ControlWidget::ControlWidget(QWidget *parent)  // FIX: constructor implementation
    : QWidget(parent)  // FIX: call base constructor
{
    setupUi();  // FIX: initialize UI components
}

ControlWidget::~ControlWidget()  // FIX: destructor implementation
{
    // FIX: Qt will auto-delete child objects
}

void ControlWidget::setupUi()  // FIX: UI initialization implementation with 30+ parameters
{
    // FIX: Create scroll area for long parameter list
    QScrollArea *scrollArea = new QScrollArea(this);  // FIX: create scroll area
    scrollArea->setWidgetResizable(true);  // FIX: auto-resize content
    scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);  // FIX: disable horizontal scroll

    // FIX: Create container widget for scroll content
    QWidget *scrollContent = new QWidget();  // FIX: scroll content widget
    QVBoxLayout *scrollLayout = new QVBoxLayout(scrollContent);  // FIX: vertical layout for sections

    // =========================
    // FIX: SECTION 1: Basic Parameters (Always Visible)
    // =========================
    QGroupBox *basicGroup = new QGroupBox("Basic Parameters", scrollContent);  // FIX: basic section
    basicGroup->setCheckable(false);  // FIX: always expanded
    QFormLayout *basicForm = new QFormLayout(basicGroup);  // FIX: form layout

    // FIX: Symbol input
    symbolEdit = new QLineEdit();  // FIX: create line edit
    symbolEdit->setText("BTCUSDT");  // FIX: default symbol
    symbolEdit->setPlaceholderText("Trading pair (e.g., BTCUSDT)");  // FIX: placeholder
    basicForm->addRow("Symbol:", symbolEdit);  // FIX: add to form

    // FIX: Timeframe selector
    timeframeCombo = new QComboBox();  // FIX: create combo box
    timeframeCombo->addItems({"1m", "5m", "15m", "30m", "1h", "4h", "1d"});  // FIX: timeframe options
    timeframeCombo->setCurrentText("1h");  // FIX: default 1h
    basicForm->addRow("Timeframe:", timeframeCombo);  // FIX: add to form

    // FIX: Forecast horizon
    horizonSpin = new QSpinBox();  // FIX: create spin box
    horizonSpin->setRange(1, 100);  // FIX: range 1-100
    horizonSpin->setValue(1);  // FIX: default 1
    basicForm->addRow("Forecast Horizon:", horizonSpin);  // FIX: add to form

    // FIX: CV enable checkbox
    cvCheckbox = new QCheckBox("Enable Cross-Validation");  // FIX: create checkbox
    cvCheckbox->setChecked(true);  // FIX: default enabled
    basicForm->addRow(cvCheckbox);  // FIX: add to form

    // FIX: CV folds spinner
    cvFoldsSpin = new QSpinBox();  // FIX: create spin box
    cvFoldsSpin->setRange(1, 20);  // FIX: range 1-20
    cvFoldsSpin->setValue(5);  // FIX: default 5
    cvFoldsSpin->setEnabled(true);  // FIX: enabled by default
    basicForm->addRow("CV Folds:", cvFoldsSpin);  // FIX: add to form

    // FIX: Connect CV checkbox to enable/disable folds spinner
    connect(cvCheckbox, &QCheckBox::toggled, [this](bool checked) {
        cvFoldsSpin->setEnabled(checked);  // FIX: enable folds only when CV checked
    });

    scrollLayout->addWidget(basicGroup);  // FIX: add basic section

    // =========================
    // FIX: SECTION 2: Advanced Training Parameters (Collapsible)
    // =========================
    QGroupBox *advancedGroup = new QGroupBox("Advanced Training Parameters", scrollContent);  // FIX: advanced section
    advancedGroup->setCheckable(true);  // FIX: collapsible
    advancedGroup->setChecked(false);  // FIX: collapsed by default
    QFormLayout *advancedForm = new QFormLayout(advancedGroup);  // FIX: form layout

    // FIX: Batch size
    batchSizeSpin = new QSpinBox();  // FIX: create spin box
    batchSizeSpin->setRange(1, 4096);  // FIX: range 1-4096
    batchSizeSpin->setValue(512);  // FIX: default 512
    advancedForm->addRow("Batch Size:", batchSizeSpin);  // FIX: add to form

    // FIX: Epochs
    epochsSpin = new QSpinBox();  // FIX: create spin box
    epochsSpin->setRange(1, 1000);  // FIX: range 1-1000
    epochsSpin->setValue(200);  // FIX: default 200
    advancedForm->addRow("Epochs:", epochsSpin);  // FIX: add to form

    // FIX: Learning rate
    lrSpin = new QDoubleSpinBox();  // FIX: create double spin box
    lrSpin->setRange(0.00001, 0.1);  // FIX: range 1e-5 to 0.1
    lrSpin->setDecimals(5);  // FIX: 5 decimal places
    lrSpin->setSingleStep(0.0001);  // FIX: step size
    lrSpin->setValue(0.001);  // FIX: default 0.001
    advancedForm->addRow("Learning Rate:", lrSpin);  // FIX: add to form

    // FIX: Weight decay
    weightDecaySpin = new QDoubleSpinBox();  // FIX: create double spin box
    weightDecaySpin->setRange(0.0, 0.1);  // FIX: range 0.0-0.1
    weightDecaySpin->setDecimals(3);  // FIX: 3 decimal places
    weightDecaySpin->setSingleStep(0.001);  // FIX: step size
    weightDecaySpin->setValue(0.01);  // FIX: default 0.01
    advancedForm->addRow("Weight Decay:", weightDecaySpin);  // FIX: add to form

    // FIX: Gradient clip norm
    gradClipNormSpin = new QDoubleSpinBox();  // FIX: create double spin box
    gradClipNormSpin->setRange(0.0, 10.0);  // FIX: range 0.0-10.0
    gradClipNormSpin->setDecimals(2);  // FIX: 2 decimal places
    gradClipNormSpin->setSingleStep(0.1);  // FIX: step size
    gradClipNormSpin->setValue(1.0);  // FIX: default 1.0
    advancedForm->addRow("Gradient Clip Norm:", gradClipNormSpin);  // FIX: add to form

    // FIX: Optimizer
    optimizerCombo = new QComboBox();  // FIX: create combo box
    optimizerCombo->addItems({"adamw", "adam", "sgd"});  // FIX: optimizer options
    optimizerCombo->setCurrentText("adamw");  // FIX: default adamw
    advancedForm->addRow("Optimizer:", optimizerCombo);  // FIX: add to form

    // FIX: Scheduler
    schedulerCombo = new QComboBox();  // FIX: create combo box
    schedulerCombo->addItems({"onecycle", "cosine", "plateau"});  // FIX: scheduler options
    schedulerCombo->setCurrentText("onecycle");  // FIX: default onecycle
    advancedForm->addRow("Scheduler:", schedulerCombo);  // FIX: add to form

    // FIX: OneCycle warmup percentage
    onecycleWarmupPctSpin = new QDoubleSpinBox();  // FIX: create double spin box
    onecycleWarmupPctSpin->setRange(0.0, 1.0);  // FIX: range 0.0-1.0
    onecycleWarmupPctSpin->setDecimals(2);  // FIX: 2 decimal places
    onecycleWarmupPctSpin->setSingleStep(0.01);  // FIX: step size
    onecycleWarmupPctSpin->setValue(0.15);  // FIX: default 0.15
    advancedForm->addRow("OneCycle Warmup %:", onecycleWarmupPctSpin);  // FIX: add to form

    // FIX: Regression weight
    regWeightSpin = new QDoubleSpinBox();  // FIX: create double spin box
    regWeightSpin->setRange(0.0, 10.0);  // FIX: range 0.0-10.0
    regWeightSpin->setDecimals(2);  // FIX: 2 decimal places
    regWeightSpin->setSingleStep(0.1);  // FIX: step size
    regWeightSpin->setValue(1.0);  // FIX: default 1.0
    advancedForm->addRow("Regression Weight:", regWeightSpin);  // FIX: add to form

    // FIX: Classification weight
    clsWeightSpin = new QDoubleSpinBox();  // FIX: create double spin box
    clsWeightSpin->setRange(0.0, 10.0);  // FIX: range 0.0-10.0
    clsWeightSpin->setDecimals(2);  // FIX: 2 decimal places
    clsWeightSpin->setSingleStep(0.1);  // FIX: step size
    clsWeightSpin->setValue(4.0);  // FIX: default 4.0
    advancedForm->addRow("Classification Weight:", clsWeightSpin);  // FIX: add to form

    // FIX: Uncertainty weight
    uncWeightSpin = new QDoubleSpinBox();  // FIX: create double spin box
    uncWeightSpin->setRange(0.0, 10.0);  // FIX: range 0.0-10.0
    uncWeightSpin->setDecimals(2);  // FIX: 2 decimal places
    uncWeightSpin->setSingleStep(0.1);  // FIX: step size
    uncWeightSpin->setValue(0.2);  // FIX: default 0.2
    advancedForm->addRow("Uncertainty Weight:", uncWeightSpin);  // FIX: add to form

    // FIX: Sign hinge weight
    signHingeWeightSpin = new QDoubleSpinBox();  // FIX: create double spin box
    signHingeWeightSpin->setRange(0.0, 10.0);  // FIX: range 0.0-10.0
    signHingeWeightSpin->setDecimals(2);  // FIX: 2 decimal places
    signHingeWeightSpin->setSingleStep(0.1);  // FIX: step size
    signHingeWeightSpin->setValue(0.5);  // FIX: default 0.5
    advancedForm->addRow("Sign Hinge Weight:", signHingeWeightSpin);  // FIX: add to form

    // FIX: Early stop patience
    earlyStopPatienceSpin = new QSpinBox();  // FIX: create spin box
    earlyStopPatienceSpin->setRange(1, 100);  // FIX: range 1-100
    earlyStopPatienceSpin->setValue(25);  // FIX: default 25
    advancedForm->addRow("Early Stop Patience:", earlyStopPatienceSpin);  // FIX: add to form

    // FIX: Random seed
    seedSpin = new QSpinBox();  // FIX: create spin box
    seedSpin->setRange(0, 9999);  // FIX: range 0-9999
    seedSpin->setValue(42);  // FIX: default 42
    advancedForm->addRow("Random Seed:", seedSpin);  // FIX: add to form

    scrollLayout->addWidget(advancedGroup);  // FIX: add advanced section

    // =========================
    // FIX: SECTION 3: Model Architecture Parameters (Collapsible)
    // =========================
    QGroupBox *modelGroup = new QGroupBox("Model Architecture Parameters", scrollContent);  // FIX: model section
    modelGroup->setCheckable(true);  // FIX: collapsible
    modelGroup->setChecked(false);  // FIX: collapsed by default
    QFormLayout *modelForm = new QFormLayout(modelGroup);  // FIX: form layout

    // FIX: Hermite degree
    hermiteDegreeSpin = new QSpinBox();  // FIX: create spin box
    hermiteDegreeSpin->setRange(3, 10);  // FIX: range 3-10
    hermiteDegreeSpin->setValue(5);  // FIX: default 5
    modelForm->addRow("Hermite Degree:", hermiteDegreeSpin);  // FIX: add to form

    // FIX: Hermite maps A
    hermiteMapsASpin = new QSpinBox();  // FIX: create spin box
    hermiteMapsASpin->setRange(1, 20);  // FIX: range 1-20
    hermiteMapsASpin->setValue(6);  // FIX: default 6
    modelForm->addRow("Hermite Maps A:", hermiteMapsASpin);  // FIX: add to form

    // FIX: Hermite maps B
    hermiteMapsBSpin = new QSpinBox();  // FIX: create spin box
    hermiteMapsBSpin->setRange(1, 20);  // FIX: range 1-20
    hermiteMapsBSpin->setValue(3);  // FIX: default 3
    modelForm->addRow("Hermite Maps B:", hermiteMapsBSpin);  // FIX: add to form

    // FIX: Hermite hidden dim
    hermiteHiddenDimSpin = new QSpinBox();  // FIX: create spin box
    hermiteHiddenDimSpin->setRange(16, 512);  // FIX: range 16-512
    hermiteHiddenDimSpin->setValue(64);  // FIX: default 64
    modelForm->addRow("Hermite Hidden Dim:", hermiteHiddenDimSpin);  // FIX: add to form

    // FIX: Dropout
    dropoutSpin = new QDoubleSpinBox();  // FIX: create double spin box
    dropoutSpin->setRange(0.0, 0.9);  // FIX: range 0.0-0.9
    dropoutSpin->setDecimals(2);  // FIX: 2 decimal places
    dropoutSpin->setSingleStep(0.01);  // FIX: step size
    dropoutSpin->setValue(0.15);  // FIX: default 0.15
    modelForm->addRow("Dropout:", dropoutSpin);  // FIX: add to form

    // FIX: Probability source
    probabilitySourceCombo = new QComboBox();  // FIX: create combo box
    probabilitySourceCombo->addItems({"cdf", "logit"});  // FIX: probability source options
    probabilitySourceCombo->setCurrentText("cdf");  // FIX: default cdf
    modelForm->addRow("Probability Source:", probabilitySourceCombo);  // FIX: add to form

    // FIX: LSTM hidden units
    lstmHiddenUnitsSpin = new QSpinBox();  // FIX: create spin box
    lstmHiddenUnitsSpin->setRange(16, 256);  // FIX: range 16-256
    lstmHiddenUnitsSpin->setValue(48);  // FIX: default 48
    lstmHiddenUnitsSpin->setEnabled(false);  // FIX: disabled by default (LSTM off)
    modelForm->addRow("LSTM Hidden Units:", lstmHiddenUnitsSpin);  // FIX: add to form

    // FIX: Use LSTM checkbox
    useLstmCheckbox = new QCheckBox("Enable LSTM");  // FIX: create checkbox
    useLstmCheckbox->setChecked(false);  // FIX: default disabled
    modelForm->addRow(useLstmCheckbox);  // FIX: add to form

    // FIX: Connect LSTM checkbox to enable/disable LSTM units spinner
    connect(useLstmCheckbox, &QCheckBox::toggled, [this](bool checked) {
        lstmHiddenUnitsSpin->setEnabled(checked);  // FIX: enable units only when LSTM checked
    });

    scrollLayout->addWidget(modelGroup);  // FIX: add model section

    // =========================
    // FIX: SECTION 4: Evaluation Parameters (Collapsible)
    // =========================
    QGroupBox *evalGroup = new QGroupBox("Evaluation Parameters", scrollContent);  // FIX: evaluation section
    evalGroup->setCheckable(true);  // FIX: collapsible
    evalGroup->setChecked(false);  // FIX: collapsed by default
    QFormLayout *evalForm = new QFormLayout(evalGroup);  // FIX: form layout

    // FIX: Calibration method
    calibrationMethodCombo = new QComboBox();  // FIX: create combo box
    calibrationMethodCombo->addItems({"abs", "std_gauss"});  // FIX: calibration methods
    calibrationMethodCombo->setCurrentText("std_gauss");  // FIX: default std_gauss
    evalForm->addRow("Calibration Method:", calibrationMethodCombo);  // FIX: add to form

    // FIX: Confidence margin
    confidenceMarginSpin = new QDoubleSpinBox();  // FIX: create double spin box
    confidenceMarginSpin->setRange(0.0, 0.5);  // FIX: range 0.0-0.5
    confidenceMarginSpin->setDecimals(3);  // FIX: 3 decimal places
    confidenceMarginSpin->setSingleStep(0.001);  // FIX: step size
    confidenceMarginSpin->setValue(0.04);  // FIX: default 0.04
    evalForm->addRow("Confidence Margin:", confidenceMarginSpin);  // FIX: add to form

    // FIX: Kelly clip
    kellyClipSpin = new QDoubleSpinBox();  // FIX: create double spin box
    kellyClipSpin->setRange(0.0, 2.0);  // FIX: range 0.0-2.0
    kellyClipSpin->setDecimals(2);  // FIX: 2 decimal places
    kellyClipSpin->setSingleStep(0.1);  // FIX: step size
    kellyClipSpin->setValue(1.0);  // FIX: default 1.0
    evalForm->addRow("Kelly Clip:", kellyClipSpin);  // FIX: add to form

    // FIX: Conformal p_min
    conformalPMinSpin = new QDoubleSpinBox();  // FIX: create double spin box
    conformalPMinSpin->setRange(0.0, 1.0);  // FIX: range 0.0-1.0
    conformalPMinSpin->setDecimals(2);  // FIX: 2 decimal places
    conformalPMinSpin->setSingleStep(0.01);  // FIX: step size
    conformalPMinSpin->setValue(0.10);  // FIX: default 0.10
    evalForm->addRow("Conformal p_min:", conformalPMinSpin);  // FIX: add to form

    // FIX: Use Kelly position checkbox
    useKellyPositionCheckbox = new QCheckBox("Use Kelly Position");  // FIX: create checkbox
    useKellyPositionCheckbox->setChecked(true);  // FIX: default enabled
    evalForm->addRow(useKellyPositionCheckbox);  // FIX: add to form

    // FIX: Use confidence margin checkbox
    useConfidenceMarginCheckbox = new QCheckBox("Use Confidence Margin");  // FIX: create checkbox
    useConfidenceMarginCheckbox->setChecked(true);  // FIX: default enabled
    evalForm->addRow(useConfidenceMarginCheckbox);  // FIX: add to form

    // FIX: Use conformal filter checkbox
    useConformalFilterCheckbox = new QCheckBox("Use Conformal Filter");  // FIX: create checkbox
    useConformalFilterCheckbox->setChecked(true);  // FIX: default enabled
    evalForm->addRow(useConformalFilterCheckbox);  // FIX: add to form

    scrollLayout->addWidget(evalGroup);  // FIX: add evaluation section

    // =========================
    // FIX: BUTTONS (Always Visible at Bottom)
    // =========================
    QHBoxLayout *buttonLayout = new QHBoxLayout();  // FIX: create horizontal layout

    // FIX: Start button
    startButton = new QPushButton("Start Training");  // FIX: create start button
    startButton->setStyleSheet("background-color: #00aa00; color: white; font-weight: bold;");  // FIX: green button
    connect(startButton, &QPushButton::clicked, this, &ControlWidget::onStartButtonClicked);  // FIX: connect signal
    buttonLayout->addWidget(startButton);  // FIX: add to layout

    // FIX: Stop button
    stopButton = new QPushButton("Stop Training");  // FIX: create stop button
    stopButton->setStyleSheet("background-color: #aa0000; color: white; font-weight: bold;");  // FIX: red button
    stopButton->setEnabled(false);  // FIX: disabled by default
    connect(stopButton, &QPushButton::clicked, this, &ControlWidget::onStopButtonClicked);  // FIX: connect signal
    buttonLayout->addWidget(stopButton);  // FIX: add to layout

    scrollLayout->addLayout(buttonLayout);  // FIX: add button layout
    scrollLayout->addStretch();  // FIX: add stretch to push controls to top

    // FIX: Set scroll content and main layout
    scrollArea->setWidget(scrollContent);  // FIX: set scroll content
    QVBoxLayout *mainLayout = new QVBoxLayout(this);  // FIX: create main layout
    mainLayout->setContentsMargins(0, 0, 0, 0);  // FIX: remove margins
    mainLayout->addWidget(scrollArea);  // FIX: add scroll area
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

QJsonObject ControlWidget::getAllParameters() const  // FIX: collect all 30+ parameters as JSON
{
    QJsonObject params;  // FIX: create JSON object

    // FIX: Basic parameters
    params["symbol"] = symbolEdit->text();  // FIX: symbol
    params["timeframe"] = timeframeCombo->currentText();  // FIX: timeframe
    params["forecast_horizon"] = horizonSpin->value();  // FIX: horizon
    params["use_cv"] = cvCheckbox->isChecked();  // FIX: CV enabled
    params["cv_folds"] = cvFoldsSpin->value();  // FIX: CV folds

    // FIX: Advanced training parameters
    params["batch_size"] = batchSizeSpin->value();  // FIX: batch size
    params["epochs"] = epochsSpin->value();  // FIX: epochs
    params["lr"] = lrSpin->value();  // FIX: learning rate
    params["weight_decay"] = weightDecaySpin->value();  // FIX: weight decay
    params["grad_clip_norm"] = gradClipNormSpin->value();  // FIX: gradient clip norm
    params["optimizer"] = optimizerCombo->currentText();  // FIX: optimizer
    params["scheduler"] = schedulerCombo->currentText();  // FIX: scheduler
    params["onecycle_warmup_pct"] = onecycleWarmupPctSpin->value();  // FIX: onecycle warmup
    params["reg_weight"] = regWeightSpin->value();  // FIX: regression weight
    params["cls_weight"] = clsWeightSpin->value();  // FIX: classification weight
    params["unc_weight"] = uncWeightSpin->value();  // FIX: uncertainty weight
    params["sign_hinge_weight"] = signHingeWeightSpin->value();  // FIX: sign hinge weight
    params["early_stop_patience"] = earlyStopPatienceSpin->value();  // FIX: early stop patience
    params["seed"] = seedSpin->value();  // FIX: random seed

    // FIX: Model architecture parameters
    params["hermite_degree"] = hermiteDegreeSpin->value();  // FIX: hermite degree
    params["hermite_maps_a"] = hermiteMapsASpin->value();  // FIX: hermite maps A
    params["hermite_maps_b"] = hermiteMapsBSpin->value();  // FIX: hermite maps B
    params["hermite_hidden_dim"] = hermiteHiddenDimSpin->value();  // FIX: hermite hidden dim
    params["dropout"] = dropoutSpin->value();  // FIX: dropout
    params["probability_source"] = probabilitySourceCombo->currentText();  // FIX: probability source
    params["lstm_hidden_units"] = lstmHiddenUnitsSpin->value();  // FIX: LSTM hidden units
    params["use_lstm"] = useLstmCheckbox->isChecked();  // FIX: use LSTM

    // FIX: Evaluation parameters
    params["calibration_method"] = calibrationMethodCombo->currentText();  // FIX: calibration method
    params["confidence_margin"] = confidenceMarginSpin->value();  // FIX: confidence margin
    params["kelly_clip"] = kellyClipSpin->value();  // FIX: Kelly clip
    params["conformal_p_min"] = conformalPMinSpin->value();  // FIX: conformal p_min
    params["use_kelly_position"] = useKellyPositionCheckbox->isChecked();  // FIX: use Kelly position
    params["use_confidence_margin"] = useConfidenceMarginCheckbox->isChecked();  // FIX: use confidence margin
    params["use_conformal_filter"] = useConformalFilterCheckbox->isChecked();  // FIX: use conformal filter

    return params;  // FIX: return JSON object with all parameters
}
