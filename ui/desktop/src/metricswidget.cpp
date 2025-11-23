// FIX: MetricsWidget implementation for displaying prediction metrics per spec

#include "metricswidget.h"  // FIX: metrics widget header
#include <QVBoxLayout>  // FIX: vertical layout
#include <QGroupBox>  // FIX: group box for organizing metrics

MetricsWidget::MetricsWidget(QWidget *parent)  // FIX: constructor implementation
    : QWidget(parent)  // FIX: call base constructor
{
    setupUi();  // FIX: initialize UI components
}

MetricsWidget::~MetricsWidget()  // FIX: destructor implementation
{
    // FIX: Qt will auto-delete child objects
}

void MetricsWidget::setupUi()  // FIX: UI initialization implementation
{
    // FIX: Create group box for metrics
    QGroupBox *metricsGroup = new QGroupBox("Prediction Metrics", this);  // FIX: create group box
    QGridLayout *gridLayout = new QGridLayout(metricsGroup);  // FIX: create grid layout

    // FIX: Create metric labels per spec
    int row = 0;  // FIX: row counter
    gridLayout->addWidget(new QLabel("AUC:"), row, 0);  // FIX: AUC label
    aucLabel = new QLabel("0.0000");  // FIX: AUC value label
    gridLayout->addWidget(aucLabel, row++, 1);  // FIX: add to grid

    gridLayout->addWidget(new QLabel("DirAcc:"), row, 0);  // FIX: directional accuracy label
    dirAccLabel = new QLabel("0.0000");  // FIX: DirAcc value label
    gridLayout->addWidget(dirAccLabel, row++, 1);  // FIX: add to grid

    gridLayout->addWidget(new QLabel("Brier:"), row, 0);  // FIX: Brier label
    brierLabel = new QLabel("1.0000");  // FIX: Brier value label
    gridLayout->addWidget(brierLabel, row++, 1);  // FIX: add to grid

    gridLayout->addWidget(new QLabel("ECE:"), row, 0);  // FIX: ECE label
    eceLabel = new QLabel("1.0000");  // FIX: ECE value label
    gridLayout->addWidget(eceLabel, row++, 1);  // FIX: add to grid

    gridLayout->addWidget(new QLabel("NLL:"), row, 0);  // FIX: NLL label
    nllLabel = new QLabel("0.0000");  // FIX: NLL value label
    gridLayout->addWidget(nllLabel, row++, 1);  // FIX: add to grid

    gridLayout->addWidget(new QLabel("MZ Intercept:"), row, 0);  // FIX: MZ intercept label
    mzInterceptLabel = new QLabel("0.0000");  // FIX: MZ intercept value label
    gridLayout->addWidget(mzInterceptLabel, row++, 1);  // FIX: add to grid

    gridLayout->addWidget(new QLabel("MZ Slope:"), row, 0);  // FIX: MZ slope label
    mzSlopeLabel = new QLabel("1.0000");  // FIX: MZ slope value label
    gridLayout->addWidget(mzSlopeLabel, row++, 1);  // FIX: add to grid

    gridLayout->addWidget(new QLabel("MZ F p-value:"), row, 0);  // FIX: MZ F-test p-value label
    mzFpLabel = new QLabel("1.0000");  // FIX: MZ F p-value value label
    gridLayout->addWidget(mzFpLabel, row++, 1);  // FIX: add to grid

    gridLayout->addWidget(new QLabel("PIT KS p-value:"), row, 0);  // FIX: PIT KS p-value label
    pitKsLabel = new QLabel("1.0000");  // FIX: PIT KS p-value value label
    gridLayout->addWidget(pitKsLabel, row++, 1);  // FIX: add to grid

    gridLayout->addWidget(new QLabel("Conformal Coverage:"), row, 0);  // FIX: conformal coverage label
    conformalCoverageLabel = new QLabel("0.0000");  // FIX: conformal coverage value label
    gridLayout->addWidget(conformalCoverageLabel, row++, 1);  // FIX: add to grid

    gridLayout->addWidget(new QLabel("Conformal Width:"), row, 0);  // FIX: conformal width label
    conformalWidthLabel = new QLabel("0.0000");  // FIX: conformal width value label
    gridLayout->addWidget(conformalWidthLabel, row++, 1);  // FIX: add to grid

    // FIX: Set main layout
    QVBoxLayout *mainLayout = new QVBoxLayout(this);  // FIX: create vertical layout
    mainLayout->addWidget(metricsGroup);  // FIX: add metrics group
    mainLayout->addStretch();  // FIX: add stretch to push metrics to top
}

void MetricsWidget::updateMetrics(const QJsonObject &metrics)  // FIX: update metrics implementation
{
    // FIX: Update metric labels from JSON per spec
    if (metrics.contains("val_auc"))  // FIX: check for AUC
        aucLabel->setText(QString::number(metrics["val_auc"].toDouble(), 'f', 4));  // FIX: update AUC label

    if (metrics.contains("DirAcc"))  // FIX: check for directional accuracy
        dirAccLabel->setText(QString::number(metrics["DirAcc"].toDouble(), 'f', 4));  // FIX: update DirAcc label

    if (metrics.contains("val_brier"))  // FIX: check for Brier
        brierLabel->setText(QString::number(metrics["val_brier"].toDouble(), 'f', 4));  // FIX: update Brier label

    if (metrics.contains("ECE"))  // FIX: check for ECE
        eceLabel->setText(QString::number(metrics["ECE"].toDouble(), 'f', 4));  // FIX: update ECE label

    if (metrics.contains("val_nll"))  // FIX: check for NLL
        nllLabel->setText(QString::number(metrics["val_nll"].toDouble(), 'f', 4));  // FIX: update NLL label

    if (metrics.contains("MZ_intercept"))  // FIX: check for MZ intercept
        mzInterceptLabel->setText(QString::number(metrics["MZ_intercept"].toDouble(), 'f', 4));  // FIX: update MZ intercept label

    if (metrics.contains("MZ_slope"))  // FIX: check for MZ slope
        mzSlopeLabel->setText(QString::number(metrics["MZ_slope"].toDouble(), 'f', 4));  // FIX: update MZ slope label

    if (metrics.contains("MZ_F_p"))  // FIX: check for MZ F p-value
        mzFpLabel->setText(QString::number(metrics["MZ_F_p"].toDouble(), 'f', 4));  // FIX: update MZ F p-value label

    if (metrics.contains("PIT_KS_p"))  // FIX: check for PIT KS p-value
        pitKsLabel->setText(QString::number(metrics["PIT_KS_p"].toDouble(), 'f', 4));  // FIX: update PIT KS p-value label

    if (metrics.contains("conformal_coverage"))  // FIX: check for conformal coverage
        conformalCoverageLabel->setText(QString::number(metrics["conformal_coverage"].toDouble(), 'f', 4));  // FIX: update conformal coverage label

    if (metrics.contains("conformal_width"))  // FIX: check for conformal width
        conformalWidthLabel->setText(QString::number(metrics["conformal_width"].toDouble(), 'f', 4));  // FIX: update conformal width label

    // FIX: Update sparkline history (simplified skeleton)
    if (metrics.contains("val_auc"))  // FIX: check for AUC
        aucHistory.append(metrics["val_auc"].toDouble());  // FIX: append to AUC history

    if (metrics.contains("val_nll"))  // FIX: check for NLL
        nllHistory.append(metrics["val_nll"].toDouble());  // FIX: append to NLL history
}

void MetricsWidget::resetMetrics()  // FIX: reset metrics implementation
{
    // FIX: Reset all metric labels to defaults
    aucLabel->setText("0.0000");  // FIX: reset AUC
    dirAccLabel->setText("0.0000");  // FIX: reset DirAcc
    brierLabel->setText("1.0000");  // FIX: reset Brier
    eceLabel->setText("1.0000");  // FIX: reset ECE
    nllLabel->setText("0.0000");  // FIX: reset NLL
    mzInterceptLabel->setText("0.0000");  // FIX: reset MZ intercept
    mzSlopeLabel->setText("1.0000");  // FIX: reset MZ slope
    mzFpLabel->setText("1.0000");  // FIX: reset MZ F p-value
    pitKsLabel->setText("1.0000");  // FIX: reset PIT KS p-value
    conformalCoverageLabel->setText("0.0000");  // FIX: reset conformal coverage
    conformalWidthLabel->setText("0.0000");  // FIX: reset conformal width

    // FIX: Clear sparkline history
    aucHistory.clear();  // FIX: clear AUC history
    nllHistory.clear();  // FIX: clear NLL history
}
