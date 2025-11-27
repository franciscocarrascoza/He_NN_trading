// FIX: MetricsWidget header for displaying prediction metrics with sparklines per spec

#ifndef METRICSWIDGET_H  // FIX: header guard
#define METRICSWIDGET_H

#include <QWidget>  // FIX: Qt widget base class
#include <QLabel>  // FIX: Qt label for metrics display
#include <QGridLayout>  // FIX: grid layout for metrics
#include <QJsonObject>  // FIX: JSON object for parsing updates
#include <QVector>  // FIX: vector for sparkline data

class MetricsWidget : public QWidget  // FIX: MetricsWidget class per spec
{
    Q_OBJECT  // FIX: Qt meta-object macro

public:
    explicit MetricsWidget(QWidget *parent = nullptr);  // FIX: constructor
    ~MetricsWidget();  // FIX: destructor

public slots:
    void updateMetrics(const QJsonObject &metrics);  // FIX: update metrics from WebSocket message
    void resetMetrics();  // FIX: reset metrics to defaults

private:
    void setupUi();  // FIX: initialize UI components

    // FIX: Metric labels per spec
    QLabel *aucLabel;  // FIX: AUC label
    QLabel *dirAccLabel;  // FIX: directional accuracy label
    QLabel *brierLabel;  // FIX: Brier score label
    QLabel *eceLabel;  // FIX: ECE label
    QLabel *nllLabel;  // FIX: NLL label
    QLabel *mzInterceptLabel;  // FIX: MZ intercept label
    QLabel *mzSlopeLabel;  // FIX: MZ slope label
    QLabel *mzFpLabel;  // FIX: MZ F-test p-value label
    QLabel *pitKsLabel;  // FIX: PIT KS p-value label
    QLabel *conformalCoverageLabel;  // FIX: conformal coverage label
    QLabel *conformalWidthLabel;  // FIX: conformal width label

    // FIX: Sparkline data (simplified - would use custom QWidget for actual sparklines)
    QVector<double> aucHistory;  // FIX: AUC history for sparkline
    QVector<double> nllHistory;  // FIX: NLL history for sparkline
};

#endif  // FIX: close header guard
