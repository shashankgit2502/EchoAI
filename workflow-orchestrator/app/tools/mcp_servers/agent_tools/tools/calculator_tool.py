from app.core.calculator.models import CalculatorInput
from app.core.calculator.service import CalculatorService

calculator_service = CalculatorService()


async def handle_calculator(arguments: dict):
    data = CalculatorInput(**arguments)
    result = calculator_service.calculate(data)
    return result.model_dump()
