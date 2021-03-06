import smtplib
from email.header import Header
from email.mime.text import MIMEText

sender = 'hongbo'
receivers = ['hongboal@qq.com']  # 接收邮箱

message = MIMEText('Python 邮件发送测试...', 'plain', 'utf-8')
message['From'] = Header("Python学习与数据挖掘", 'utf-8')  # 发送者
message['To'] = Header("测试", 'utf-8')  # 接收者

subject = 'Python SMTP 邮件测试'
message['Subject'] = Header(subject, 'utf-8')

try:
    smtpObj = smtplib.SMTP('127.0.0.1')
    smtpObj.sendmail(sender, receivers, message.as_string())
    print("邮件发送成功")
except smtplib.SMTPException:
    print("Error: 无法发送邮件")
