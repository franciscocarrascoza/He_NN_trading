// FIX: ControlWidget header for training control panel (no trading UI) per spec

#ifndef CONTROLWIDGET_H  // FIX: header guard
#define CONTROLWIDGET_H

#include <QWidget>  // FIX: Qt widget base class
#include <QPushButton>  // FIX: push button
#include <QComboBox>  // FIX: combo box for selections
#include <QSpinBox>  // FIX: spin box for numeric inputs
#include <QDoubleSpinBox>  // FIX: double spin box for floating-point inputs
#include <QCheckBox>  // FIX: checkbox for toggles
#include <QLineEdit>  // FIX: line edit for text inputs
#include <QJsonObject>  // FIX: JSON object for parameter collection

class ControlWidget : public QWidget  // FIX: ControlWidget class per spec
{
    Q_OBJECT  // FIX: Qt meta-object macro

public:
    explicit ControlWidget(QWidget *parent = nullptr);  // FIX: constructor
    ~ControlWidget();  // FIX: destructor

    // FIX: Getter methods for all parameters
    QJsonObject getAllParameters() const;  // FIX: collect all 30+ parameters as JSON

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

    // FIX: Basic parameters (always visible)
    QLineEdit *symbolEdit;  // FIX: symbol text input (BTCUSDT)
    QComboBox *timeframeCombo;  // FIX: timeframe selector per spec
    QSpinBox *horizonSpin;  // FIX: forecast horizon spinner per spec
    QCheckBox *cvCheckbox;  // FIX: CV enable checkbox per spec
    QSpinBox *cvFoldsSpin;  // FIX: CV folds spinner (1-20, default 5)

    // FIX: Advanced training parameters (collapsible)
    QSpinBox *batchSizeSpin;  // FIX: batch size (1-4096, default 512)
    QSpinBox *epochsSpin;  // FIX: epochs (1-1000, default 200)
    QDoubleSpinBox *lrSpin;  // FIX: learning rate (1e-5 to 0.1, default 0.001)
    QDoubleSpinBox *weightDecaySpin;  // FIX: weight decay (0.0-0.1, default 0.01)
    QDoubleSpinBox *gradClipNormSpin;  // FIX: gradient clip norm (0.0-10.0, default 1.0)
    QComboBox *optimizerCombo;  // FIX: optimizer (adamw, adam, sgd)
    QComboBox *schedulerCombo;  // FIX: scheduler (onecycle, cosine, plateau)
    QDoubleSpinBox *onecycleWarmupPctSpin;  // FIX: onecycle warmup % (0.0-1.0, default 0.15)
    QDoubleSpinBox *regWeightSpin;  // FIX: regression loss weight (0.0-10.0, default 1.0)
    QDoubleSpinBox *clsWeightSpin;  // FIX: classification loss weight (0.0-10.0, default 4.0)
    QDoubleSpinBox *uncWeightSpin;  // FIX: uncertainty loss weight (0.0-10.0, default 0.2)
    QDoubleSpinBox *signHingeWeightSpin;  // FIX: sign hinge loss weight (0.0-10.0, default 0.5)
    QSpinBox *earlyStopPatienceSpin;  // FIX: early stop patience (1-100, default 25)
    QSpinBox *seedSpin;  // FIX: random seed (0-9999, default 42)

    // FIX: Model architecture parameters (collapsible)
    QSpinBox *hermiteDegreeSpin;  // FIX: hermite degree (3-10, default 5)
    QSpinBox *hermiteMapsASpin;  // FIX: hermite maps A (1-20, default 6)
    QSpinBox *hermiteMapsBSpin;  // FIX: hermite maps B (1-20, default 3)
    QSpinBox *hermiteHiddenDimSpin;  // FIX: hermite hidden dim (16-512, default 64)
    QDoubleSpinBox *dropoutSpin;  // FIX: dropout (0.0-0.9, default 0.15)
    QComboBox *probabilitySourceCombo;  // FIX: probability source (cdf, logit)
    QSpinBox *lstmHiddenUnitsSpin;  // FIX: LSTM hidden units (16-256, default 48)
    QCheckBox *useLstmCheckbox;  // FIX: use LSTM checkbox (default false)

    // FIX: Evaluation parameters (collapsible)
    QComboBox *calibrationMethodCombo;  // FIX: calibration method selector per spec
    QDoubleSpinBox *confidenceMarginSpin;  // FIX: confidence margin (0.0-0.5, default 0.04)
    QDoubleSpinBox *kellyClipSpin;  // FIX: Kelly clip (0.0-2.0, default 1.0)
    QDoubleSpinBox *conformalPMinSpin;  // FIX: conformal p_min (0.0-1.0, default 0.10)
    QCheckBox *useKellyPositionCheckbox;  // FIX: use Kelly position checkbox (default true)
    QCheckBox *useConfidenceMarginCheckbox;  // FIX: use confidence margin checkbox (default true)
    QCheckBox *useConformalFilterCheckbox;  // FIX: use conformal filter checkbox (default true)
};

#endif  // FIX: close header guard
