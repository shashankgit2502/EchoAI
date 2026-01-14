from app.core.file_reader.models import FileReaderInput
from app.core.file_reader.service import FileReaderService

service = FileReaderService()


async def handle_file_reader(arguments: dict):
    data = FileReaderInput(**arguments)
    return service.process(data)
