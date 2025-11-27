// FIX: ChartWidget implementation for OHLCV candlestick chart with overlays per spec

#include "chartwidget.h"  // FIX: chart widget header
#include <QVBoxLayout>  // FIX: vertical layout
#include <QHBoxLayout>  // FIX: horizontal layout
#include <QPushButton>  // FIX: push button
#include <QJsonObject>  // FIX: JSON object
#include <QJsonArray>  // FIX: JSON array
#include <QDateTime>  // FIX: datetime handling
#include <QFileDialog>  // FIX: file dialog
#include <QMessageBox>  // FIX: message box
#include <QtCharts/QDateTimeAxis>  // FIX: datetime axis for timestamps
#include <QtCharts/QValueAxis>  // FIX: value axis for prices
#include <QtCharts/QCandlestickSet>  // FIX: candlestick set

ChartWidget::ChartWidget(QWidget *parent)  // FIX: constructor implementation
    : QWidget(parent)  // FIX: call base constructor
    , chartView(nullptr)  // FIX: initialize chart view pointer
    , chart(nullptr)  // FIX: initialize chart pointer
    , candlestickSeries(nullptr)  // FIX: initialize candlestick series pointer
    , predictionSeries(nullptr)  // FIX: initialize prediction series pointer
    , conformalLowerSeries(nullptr)  // FIX: initialize lower conformal series pointer
    , conformalUpperSeries(nullptr)  // FIX: initialize upper conformal series pointer
{
    setupChart();  // FIX: initialize chart components
}

ChartWidget::~ChartWidget()  // FIX: destructor implementation
{
    // FIX: Qt will auto-delete child objects
}

void ChartWidget::setupChart()  // FIX: chart initialization implementation
{
    // FIX: Create chart
    chart = new QChart();  // FIX: instantiate chart
    chart->setTitle("BTCUSDT 1h with Predictions");  // FIX: set chart title per spec
    chart->setAnimationOptions(QChart::NoAnimation);  // FIX: disable animations for performance

    // FIX: Create candlestick series for OHLCV per spec
    candlestickSeries = new QCandlestickSeries();  // FIX: instantiate candlestick series
    candlestickSeries->setName("OHLCV");  // FIX: set series name
    candlestickSeries->setIncreasingColor(QColor(0, 255, 0));  // FIX: green for up candles per spec
    candlestickSeries->setDecreasingColor(QColor(255, 0, 0));  // FIX: red for down candles per spec
    chart->addSeries(candlestickSeries);  // FIX: add series to chart

    // FIX: Create prediction scatter series per spec
    predictionSeries = new QScatterSeries();  // FIX: instantiate scatter series
    predictionSeries->setName("Predictions (μ)");  // FIX: set series name
    predictionSeries->setColor(QColor(255, 170, 0));  // FIX: orange for predictions per spec
    predictionSeries->setMarkerSize(8.0);  // FIX: marker size
    chart->addSeries(predictionSeries);  // FIX: add series to chart

    // FIX: Create conformal interval series per spec
    conformalLowerSeries = new QLineSeries();  // FIX: instantiate lower line series
    conformalLowerSeries->setName("Conformal Lower");  // FIX: set series name
    conformalLowerSeries->setColor(QColor(51, 153, 255, 128));  // FIX: semi-transparent blue per spec
    chart->addSeries(conformalLowerSeries);  // FIX: add series to chart

    conformalUpperSeries = new QLineSeries();  // FIX: instantiate upper line series
    conformalUpperSeries->setName("Conformal Upper");  // FIX: set series name
    conformalUpperSeries->setColor(QColor(51, 153, 255, 128));  // FIX: semi-transparent blue per spec
    chart->addSeries(conformalUpperSeries);  // FIX: add series to chart

    // FIX: Create axes
    QDateTimeAxis *axisX = new QDateTimeAxis();  // FIX: datetime x-axis
    axisX->setFormat("yyyy-MM-dd hh:mm");  // FIX: datetime format
    axisX->setTitleText("Time");  // FIX: x-axis title
    chart->addAxis(axisX, Qt::AlignBottom);  // FIX: add bottom axis
    candlestickSeries->attachAxis(axisX);  // FIX: attach to candlestick series
    predictionSeries->attachAxis(axisX);  // FIX: attach to prediction series
    conformalLowerSeries->attachAxis(axisX);  // FIX: attach to lower conformal series
    conformalUpperSeries->attachAxis(axisX);  // FIX: attach to upper conformal series

    QValueAxis *axisY = new QValueAxis();  // FIX: value y-axis
    axisY->setTitleText("Price (USD)");  // FIX: y-axis title
    chart->addAxis(axisY, Qt::AlignLeft);  // FIX: add left axis
    candlestickSeries->attachAxis(axisY);  // FIX: attach to candlestick series
    predictionSeries->attachAxis(axisY);  // FIX: attach to prediction series
    conformalLowerSeries->attachAxis(axisY);  // FIX: attach to lower conformal series
    conformalUpperSeries->attachAxis(axisY);  // FIX: attach to upper conformal series

    // FIX: Create chart view
    chartView = new QChartView(chart, this);  // FIX: instantiate chart view
    chartView->setRenderHint(QPainter::Antialiasing);  // FIX: enable antialiasing

    // FIX: Create export buttons
    QPushButton *exportPNGBtn = new QPushButton("Export Chart (PNG)", this);  // FIX: PNG export button
    QPushButton *exportCSVBtn = new QPushButton("Export Data (CSV)", this);  // FIX: CSV export button

    // FIX: Connect export buttons
    connect(exportPNGBtn, &QPushButton::clicked, this, &ChartWidget::exportToPNG);  // FIX: connect PNG export
    connect(exportCSVBtn, &QPushButton::clicked, this, &ChartWidget::exportToCSV);  // FIX: connect CSV export

    // FIX: Create button layout
    QHBoxLayout *buttonLayout = new QHBoxLayout();  // FIX: horizontal layout for buttons
    buttonLayout->addWidget(exportPNGBtn);  // FIX: add PNG button
    buttonLayout->addWidget(exportCSVBtn);  // FIX: add CSV button
    buttonLayout->addStretch();  // FIX: push buttons to left

    // FIX: Set main layout
    QVBoxLayout *layout = new QVBoxLayout(this);  // FIX: create vertical layout
    layout->addWidget(chartView);  // FIX: add chart view to layout
    layout->addLayout(buttonLayout);  // FIX: add button layout
    layout->setContentsMargins(0, 0, 0, 0);  // FIX: remove margins
}

void ChartWidget::updateChart(const QJsonObject &data)  // FIX: update chart with new data (full replace)
{
    // FIX: Parse JSON data and completely replace chart contents
    // FIX: Expected format: {"symbol": "BTCUSDT", "timeframe": "1h", "candles": [...]}

    QString symbol = data["symbol"].toString();  // FIX: get symbol
    QString timeframe = data["timeframe"].toString();  // FIX: get timeframe
    QJsonArray candles = data["candles"].toArray();  // FIX: get candles array

    // FIX: Update chart title
    updateChartTitle(symbol, timeframe);  // FIX: update title

    // FIX: Clear existing candlesticks
    candlestickSeries->clear();  // FIX: clear series

    // FIX: Add all candles
    updateCandles(candles);  // FIX: populate with new candles
}

void ChartWidget::updateCandles(const QJsonArray &candles)  // FIX: update chart with candle array (append mode)
{
    // FIX: Parse candles array and append to candlestick series
    // FIX: Expected format: [{"timestamp": 1234567890, "open": 42000, "high": 42500, "low": 41500, "close": 42200, "volume": 123.45}, ...]

    for (const QJsonValue &candleValue : candles) {
        QJsonObject candle = candleValue.toObject();  // FIX: get candle object

        // FIX: Extract OHLCV values
        qint64 timestamp = candle["timestamp"].toInt();  // FIX: Unix timestamp
        double open = candle["open"].toDouble();  // FIX: open price
        double high = candle["high"].toDouble();  // FIX: high price
        double low = candle["low"].toDouble();  // FIX: low price
        double close = candle["close"].toDouble();  // FIX: close price
        // double volume = candle["volume"].toDouble();  // FIX: volume (not used yet)

        // FIX: Create candlestick set
        QCandlestickSet *set = new QCandlestickSet(timestamp);  // FIX: create set with timestamp
        set->setOpen(open);  // FIX: set open
        set->setHigh(high);  // FIX: set high
        set->setLow(low);  // FIX: set low
        set->setClose(close);  // FIX: set close

        // FIX: Append to series
        candlestickSeries->append(set);  // FIX: add to series
    }

    // FIX: Auto-scale axes after adding data
    if (!candles.isEmpty()) {
        chart->createDefaultAxes();  // FIX: recreate axes with proper ranges

        // FIX: Re-attach series to axes
        QList<QAbstractAxis*> axesX = chart->axes(Qt::Horizontal);  // FIX: get X axes
        QList<QAbstractAxis*> axesY = chart->axes(Qt::Vertical);  // FIX: get Y axes

        if (!axesX.isEmpty() && !axesY.isEmpty()) {
            candlestickSeries->attachAxis(axesX.first());  // FIX: attach to X axis
            candlestickSeries->attachAxis(axesY.first());  // FIX: attach to Y axis
            predictionSeries->attachAxis(axesX.first());  // FIX: attach predictions to X axis
            predictionSeries->attachAxis(axesY.first());  // FIX: attach predictions to Y axis
            conformalLowerSeries->attachAxis(axesX.first());  // FIX: attach lower band to X axis
            conformalLowerSeries->attachAxis(axesY.first());  // FIX: attach lower band to Y axis
            conformalUpperSeries->attachAxis(axesX.first());  // FIX: attach upper band to X axis
            conformalUpperSeries->attachAxis(axesY.first());  // FIX: attach upper band to Y axis
        }
    }
}

void ChartWidget::updateChartTitle(const QString &symbol, const QString &timeframe)  // FIX: update chart title
{
    chart->setTitle(QString("%1 %2 with Predictions").arg(symbol, timeframe));  // FIX: set new title
}

void ChartWidget::addPredictionOverlay(double timestamp, double mu, double lower, double upper)  // FIX: add prediction overlay
{
    // FIX: Add prediction point to scatter series per spec
    predictionSeries->append(timestamp, mu);  // FIX: append prediction point

    // FIX: Add conformal interval points per spec
    conformalLowerSeries->append(timestamp, lower);  // FIX: append lower bound
    conformalUpperSeries->append(timestamp, upper);  // FIX: append upper bound
}

void ChartWidget::exportToPNG()  // FIX: export chart to PNG file
{
    // FIX: Generate default filename with timestamp
    QString defaultName = QString("chart_%1_%2_%3.png")
        .arg(chart->title().replace(" with Predictions", "").replace(" ", "_"))  // FIX: extract symbol_timeframe
        .arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));  // FIX: timestamp

    // FIX: Open file dialog
    QString fileName = QFileDialog::getSaveFileName(
        this,  // FIX: parent widget
        "Export Chart as PNG",  // FIX: dialog title
        defaultName,  // FIX: default filename
        "PNG Image (*.png)"  // FIX: file filter
    );

    if (fileName.isEmpty()) {
        return;  // FIX: user cancelled
    }

    // FIX: Capture chart view as image
    QPixmap pixmap = chartView->grab();  // FIX: grab chart view contents

    // FIX: Save to file
    if (pixmap.save(fileName, "PNG")) {
        QMessageBox::information(this, "Export Successful",
            QString("Chart exported to:\n%1").arg(fileName));  // FIX: success message
        qInfo() << "Chart exported to PNG:" << fileName;  // FIX: log success
    } else {
        QMessageBox::critical(this, "Export Failed",
            QString("Failed to save chart to:\n%1").arg(fileName));  // FIX: error message
        qWarning() << "Failed to export chart to PNG:" << fileName;  // FIX: log error
    }
}

void ChartWidget::exportToCSV()  // FIX: export chart data to CSV file
{
    // FIX: Generate default filename with timestamp
    QString defaultName = QString("chart_data_%1_%2_%3.csv")
        .arg(chart->title().replace(" with Predictions", "").replace(" ", "_"))  // FIX: extract symbol_timeframe
        .arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));  // FIX: timestamp

    // FIX: Open file dialog
    QString fileName = QFileDialog::getSaveFileName(
        this,  // FIX: parent widget
        "Export Chart Data as CSV",  // FIX: dialog title
        defaultName,  // FIX: default filename
        "CSV File (*.csv)"  // FIX: file filter
    );

    if (fileName.isEmpty()) {
        return;  // FIX: user cancelled
    }

    // FIX: Open file for writing
    QFile file(fileName);  // FIX: create file object
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::critical(this, "Export Failed",
            QString("Failed to open file for writing:\n%1").arg(fileName));  // FIX: error message
        return;
    }

    QTextStream out(&file);  // FIX: create text stream

    // FIX: Write CSV header
    out << "timestamp,open,high,low,close,prediction_mu,conformal_lower,conformal_upper\n";  // FIX: CSV header

    // FIX: Get candlestick data
    QList<QCandlestickSet*> candleSets = candlestickSeries->sets();  // FIX: get all candlestick sets

    // FIX: Create maps for predictions indexed by timestamp
    QMap<qint64, double> predictionMap;  // FIX: map timestamp -> mu
    QMap<qint64, double> lowerBandMap;  // FIX: map timestamp -> lower
    QMap<qint64, double> upperBandMap;  // FIX: map timestamp -> upper

    // FIX: Populate prediction maps
    const QList<QPointF>& predPoints = predictionSeries->points();  // FIX: get prediction points
    for (const QPointF& point : predPoints) {
        predictionMap[static_cast<qint64>(point.x())] = point.y();  // FIX: store mu
    }

    const QList<QPointF>& lowerPoints = conformalLowerSeries->points();  // FIX: get lower band points
    for (const QPointF& point : lowerPoints) {
        lowerBandMap[static_cast<qint64>(point.x())] = point.y();  // FIX: store lower
    }

    const QList<QPointF>& upperPoints = conformalUpperSeries->points();  // FIX: get upper band points
    for (const QPointF& point : upperPoints) {
        upperBandMap[static_cast<qint64>(point.x())] = point.y();  // FIX: store upper
    }

    // FIX: Write data rows
    for (QCandlestickSet* set : candleSets) {
        qint64 timestamp = static_cast<qint64>(set->timestamp());  // FIX: get timestamp
        double open = set->open();  // FIX: get open
        double high = set->high();  // FIX: get high
        double low = set->low();  // FIX: get low
        double close = set->close();  // FIX: get close

        // FIX: Lookup prediction data for this timestamp
        QString predMu = predictionMap.contains(timestamp) ? QString::number(predictionMap[timestamp]) : "";  // FIX: get mu or empty
        QString predLower = lowerBandMap.contains(timestamp) ? QString::number(lowerBandMap[timestamp]) : "";  // FIX: get lower or empty
        QString predUpper = upperBandMap.contains(timestamp) ? QString::number(upperBandMap[timestamp]) : "";  // FIX: get upper or empty

        // FIX: Write CSV row
        out << timestamp << ","
            << open << ","
            << high << ","
            << low << ","
            << close << ","
            << predMu << ","
            << predLower << ","
            << predUpper << "\n";  // FIX: CSV row
    }

    file.close();  // FIX: close file

    // FIX: Show success message
    QMessageBox::information(this, "Export Successful",
        QString("Chart data exported to:\n%1\n\nRows: %2").arg(fileName).arg(candleSets.size()));  // FIX: success message
    qInfo() << "Chart data exported to CSV:" << fileName << "(" << candleSets.size() << "rows)";  // FIX: log success
}
