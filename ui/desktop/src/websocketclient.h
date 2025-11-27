// FIX: WebSocketClient header for backend streaming connection per spec

#ifndef WEBSOCKETCLIENT_H  // FIX: header guard
#define WEBSOCKETCLIENT_H

#include <QObject>  // FIX: Qt object base class
#include <QWebSocket>  // FIX: Qt WebSocket
#include <QString>  // FIX: Qt string

class WebSocketClient : public QObject  // FIX: WebSocketClient class per spec
{
    Q_OBJECT  // FIX: Qt meta-object macro

public:
    explicit WebSocketClient(QObject *parent = nullptr);  // FIX: constructor
    ~WebSocketClient();  // FIX: destructor

    void connectToServer(const QString &url);  // FIX: connect to backend WebSocket
    void disconnect();  // FIX: disconnect from backend
    bool isConnected() const;  // FIX: check connection status

signals:
    void connected();  // FIX: connected signal
    void disconnected();  // FIX: disconnected signal
    void messageReceived(const QString &message);  // FIX: message received signal
    void errorOccurred(const QString &error);  // FIX: error occurred signal

private slots:
    void onConnected();  // FIX: connection handler
    void onDisconnected();  // FIX: disconnection handler
    void onTextMessageReceived(const QString &message);  // FIX: message handler
    void onError(QAbstractSocket::SocketError error);  // FIX: error handler

private:
    QWebSocket *webSocket;  // FIX: WebSocket instance pointer
};

#endif  // FIX: close header guard
