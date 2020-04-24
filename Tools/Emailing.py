import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


port = 587
smtp_server = "smtp.gmail.com"
login = "electric.birder@gmail.com"  # paste your login
password = "JuLian50210809"  # paste your password
sender_email = "electric.birder@gmail.com"


class Emailer:

    def sendmail(self, recipient, subject, body, filename=None):

        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = recipient
        message["Subject"] = subject
        # Add body
        message.attach(MIMEText(body, "plain"))

        # Add attachment if provided
        if filename is not None:
            with open(filename, "rb") as attachment:
                # The content type "application/octet-stream" means that a MIME attachment is a binary file
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())

                # Encode to base64
                encoders.encode_base64(part)

                # Add header
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {filename}",
                )

                # Add attachment to your message and convert it to string
                message.attach(part)

        text = message.as_string()

        # send your email
        with smtplib.SMTP(smtp_server, port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(login, password)
            server.sendmail(
                sender_email, recipient, text
            )


# TESTING
# sender = Emailer()
# import time
# sendTo = 'kkj@jensenkk.net'
# emailSubject = "Pileated Woodpecker Seen!"
# emailContent = "Pileated Woodpecker seen at " + time.ctime()
# sender.sendmail(sendTo, emailSubject, emailContent)
#
# print("Email Sent")
