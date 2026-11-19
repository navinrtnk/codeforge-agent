"""Persist successful model response usage and optional operator-supplied USD estimates."""

import uuid
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agent.database import get_database_session
from agent.models import ModelUsage
from agent.models_api import ModelClient, ModelRequest, ModelResponse
from agent.observability import logger, request_id


class ModelPrice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_per_million: Decimal = Field(ge=0, allow_inf_nan=False)
    output_per_million: Decimal = Field(ge=0, allow_inf_nan=False)


class UsageModelClient:
    def __init__(
        self,
        client: ModelClient,
        session: Session,
        prices: dict[str, ModelPrice],
        *,
        run_id: uuid.UUID | None = None,
    ) -> None:
        self.client, self.session, self.prices = client, session, prices
        self.run_id = run_id

    async def complete(self, request: ModelRequest) -> ModelResponse:
        response = await self.client.complete(request)
        price = self.prices.get(f"{response.provider}:{response.model}")
        cost = (
            None
            if price is None
            else (
                Decimal(response.usage.input_tokens) * price.input_per_million
                + Decimal(response.usage.output_tokens) * price.output_per_million
            )
            / Decimal(1_000_000)
        )
        self.session.add(
            ModelUsage(
                agent_run_id=self.run_id,
                request_id=request_id.get(),
                provider=response.provider,
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                estimated_cost_usd=cost,
            )
        )
        # Record returned usage even if later tool execution or review validation fails.
        self.session.commit()
        logger.info("model.completed")
        return response


router = APIRouter(tags=["usage"])


@router.get("/usage")
def usage_summary(session: Annotated[Session, Depends(get_database_session)]) -> dict[str, object]:
    rows = session.execute(
        select(
            ModelUsage.provider,
            ModelUsage.model,
            func.count(ModelUsage.id),
            func.sum(ModelUsage.input_tokens),
            func.sum(ModelUsage.output_tokens),
            func.sum(ModelUsage.estimated_cost_usd),
            func.count(ModelUsage.estimated_cost_usd),
        )
        .group_by(ModelUsage.provider, ModelUsage.model)
        .order_by(ModelUsage.provider, ModelUsage.model)
    )
    return {
        "models": [
            {
                "provider": provider,
                "model": model,
                "calls": calls,
                "input_tokens": inputs,
                "output_tokens": outputs,
                "estimated_cost_usd": str(cost) if cost is not None else None,
                "unpriced_calls": calls - priced,
            }
            for provider, model, calls, inputs, outputs, cost, priced in rows
        ]
    }
