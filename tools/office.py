"""Office tools for Email and Calendar using local Outlook COM integration."""

import datetime
from typing import Any
import sys

from tools.registry import ToolResult, tool


@tool(name="read_inbox", verify=False, category="office")
async def read_inbox(count: int = 5) -> ToolResult:
    """Read recent unread emails from Microsoft Outlook.
    
    Returns sender, subject, and snippet for the top N unread emails.
    """
    if sys.platform != "win32":
        return ToolResult(success=False, error="Outlook integration is only supported on Windows.")
        
    try:
        import win32com.client
        import pythoncom
        pythoncom.CoInitialize()
        
        outlook = win32com.client.Dispatch("Outlook.Application")
        namespace = outlook.GetNamespace("MAPI")
        inbox = namespace.GetDefaultFolder(6) # 6 = Inbox
        
        messages = inbox.Items
        messages.Sort("[ReceivedTime]", True) # Sort descending
        
        results = []
        for msg in messages:
            if msg.UnRead:
                sender = getattr(msg, "SenderName", "Unknown Sender")
                subject = getattr(msg, "Subject", "No Subject")
                body = getattr(msg, "Body", "")
                snippet = body[:100].replace("\n", " ").replace("\r", "") + ("..." if len(body) > 100 else "")
                
                results.append(f"From: {sender}\nSubject: {subject}\nPreview: {snippet}")
                
                if len(results) >= count:
                    break
                    
        pythoncom.CoUninitialize()
        
        if not results:
            return ToolResult(success=True, data="No unread emails found in your inbox.")
            
        summary = f"Found {len(results)} unread emails:\n\n" + "\n\n".join(results)
        return ToolResult(success=True, data=summary)
        
    except Exception as e:
        return ToolResult(success=False, error=f"Failed to read Outlook inbox: {e}")


@tool(name="send_email", verify=True, category="office")
async def send_email(to: str, subject: str, body: str) -> ToolResult:
    """Draft and send an email via Microsoft Outlook. Requires UNLOCKED state."""
    if sys.platform != "win32":
        return ToolResult(success=False, error="Outlook integration is only supported on Windows.")
        
    try:
        import win32com.client
        import pythoncom
        pythoncom.CoInitialize()
        
        outlook = win32com.client.Dispatch("Outlook.Application")
        mail = outlook.CreateItem(0) # 0 = MailItem
        
        mail.To = to
        mail.Subject = subject
        mail.Body = body
        mail.Send()
        
        pythoncom.CoUninitialize()
        return ToolResult(success=True, data=f"Email sent successfully to {to}.")
        
    except Exception as e:
        return ToolResult(success=False, error=f"Failed to send email via Outlook: {e}")


@tool(name="get_schedule", verify=False, category="office")
async def get_schedule(days: int = 1) -> ToolResult:
    """Read upcoming calendar appointments from Microsoft Outlook."""
    if sys.platform != "win32":
        return ToolResult(success=False, error="Outlook integration is only supported on Windows.")
        
    try:
        import win32com.client
        import pythoncom
        pythoncom.CoInitialize()
        
        outlook = win32com.client.Dispatch("Outlook.Application")
        namespace = outlook.GetNamespace("MAPI")
        calendar = namespace.GetDefaultFolder(9) # 9 = Calendar
        
        appointments = calendar.Items
        appointments.IncludeRecurrences = True
        appointments.Sort("[Start]")
        
        # Filter for the next 'days' window
        begin = datetime.datetime.now()
        end = begin + datetime.timedelta(days=days)
        
        # Format strings expected by Outlook COM filter
        begin_str = begin.strftime("%m/%d/%Y %H:%M %p")
        end_str = end.strftime("%m/%d/%Y %H:%M %p")
        
        restriction = f"[Start] >= '{begin_str}' AND [Start] <= '{end_str}'"
        restricted_items = appointments.Restrict(restriction)
        
        results = []
        for appt in restricted_items:
            subject = getattr(appt, "Subject", "No Subject")
            start = getattr(appt, "Start", None)
            end_time = getattr(appt, "End", None)
            
            # Formatter
            if start and end_time:
                time_str = f"{start.strftime('%I:%M %p')} - {end_time.strftime('%I:%M %p')}"
            else:
                time_str = "Unknown time"
                
            results.append(f"• {subject} ({time_str})")
            
        pythoncom.CoUninitialize()
        
        if not results:
            return ToolResult(success=True, data=f"No appointments found for the next {days} day(s).")
            
        summary = f"Schedule for the next {days} day(s):\n" + "\n".join(results)
        return ToolResult(success=True, data=summary)
        
    except Exception as e:
        return ToolResult(success=False, error=f"Failed to read Outlook calendar: {e}")
