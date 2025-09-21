"""
Automation Rules Engine Module
Rule-based automation and intelligent decision making
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Any, Callable, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import operator
from functools import reduce


class RuleType(Enum):
    """Types of automation rules"""
    TRIGGER = "trigger"
    CONDITION = "condition"
    FILTER = "filter"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    ROUTING = "routing"
    AGGREGATION = "aggregation"
    ENRICHMENT = "enrichment"


class ConditionOperator(Enum):
    """Condition operators"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    IS_TRUE = "is_true"
    IS_FALSE = "is_false"


class ActionType(Enum):
    """Types of actions"""
    EXECUTE = "execute"
    NOTIFY = "notify"
    LOG = "log"
    STORE = "store"
    TRANSFORM = "transform"
    ROUTE = "route"
    AGGREGATE = "aggregate"
    ENRICH = "enrich"
    FILTER = "filter"
    DELAY = "delay"
    RETRY = "retry"
    ESCALATE = "escalate"
    WEBHOOK = "webhook"
    EMAIL = "email"
    SMS = "sms"


class LogicalOperator(Enum):
    """Logical operators for combining conditions"""
    AND = "and"
    OR = "or"
    NOT = "not"
    XOR = "xor"


@dataclass
class Condition:
    """Rule condition"""
    field: str
    operator: ConditionOperator
    value: Any
    case_sensitive: bool = True
    data_type: Optional[str] = None  # string, number, boolean, date
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConditionGroup:
    """Group of conditions with logical operator"""
    operator: LogicalOperator
    conditions: List[Union[Condition, 'ConditionGroup']]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """Rule action"""
    action_type: ActionType
    target: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    async_execution: bool = False
    timeout: Optional[int] = None  # seconds
    retry_on_failure: bool = False
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Rule:
    """Automation rule"""
    rule_id: str
    name: str
    description: str
    rule_type: RuleType
    conditions: Optional[ConditionGroup] = None
    actions: List[Action] = field(default_factory=list)
    priority: int = 0  # Higher value = higher priority
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleSet:
    """Collection of related rules"""
    ruleset_id: str
    name: str
    description: str
    rules: List[Rule] = field(default_factory=list)
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleExecutionContext:
    """Context for rule execution"""
    data: Dict[str, Any]
    variables: Dict[str, Any] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)  # Executed rule IDs
    results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleExecutionResult:
    """Result of rule execution"""
    rule_id: str
    triggered: bool
    conditions_met: bool
    actions_executed: List[str]
    execution_time: float
    errors: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)


class AutomationRulesEngine:
    """
    Advanced rule-based automation engine
    """
    
    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self.rulesets: Dict[str, RuleSet] = {}
        self.action_handlers: Dict[ActionType, Callable] = {}
        self.custom_functions: Dict[str, Callable] = {}
        
        # Rule execution tracking
        self.execution_history: List[RuleExecutionResult] = []
        self.active_executions: Set[str] = set()
        
        # Event queue for async processing
        self.event_queue: asyncio.Queue = asyncio.Queue()
        
        # Metrics
        self.metrics = {
            "total_rules": 0,
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0
        }
        
        self._register_default_handlers()
        self._start_engine()
    
    def _register_default_handlers(self):
        """Register default action handlers"""
        self.register_action_handler(ActionType.LOG, self._handle_log_action)
        self.register_action_handler(ActionType.TRANSFORM, self._handle_transform_action)
        self.register_action_handler(ActionType.FILTER, self._handle_filter_action)
        self.register_action_handler(ActionType.NOTIFY, self._handle_notify_action)
        self.register_action_handler(ActionType.STORE, self._handle_store_action)
        self.register_action_handler(ActionType.DELAY, self._handle_delay_action)
        self.register_action_handler(ActionType.WEBHOOK, self._handle_webhook_action)
    
    def _start_engine(self):
        """Start the rules engine"""
        asyncio.create_task(self._process_event_queue())
        asyncio.create_task(self._cleanup_history())
    
    async def _process_event_queue(self):
        """Process queued events"""
        while True:
            try:
                event = await self.event_queue.get()
                await self.process_event(event)
            except Exception as e:
                print(f"Event processing error: {e}")
    
    async def _cleanup_history(self):
        """Clean up old execution history"""
        while True:
            try:
                # Keep only last 10000 executions
                if len(self.execution_history) > 10000:
                    self.execution_history = self.execution_history[-10000:]
                
                await asyncio.sleep(3600)  # Cleanup hourly
                
            except Exception as e:
                print(f"Cleanup error: {e}")
                await asyncio.sleep(3600)
    
    def add_rule(self, rule: Rule) -> bool:
        """Add a rule to the engine"""
        if rule.rule_id in self.rules:
            return False
        
        self.rules[rule.rule_id] = rule
        self.metrics["total_rules"] += 1
        return True
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from the engine"""
        if rule_id not in self.rules:
            return False
        
        del self.rules[rule_id]
        self.metrics["total_rules"] -= 1
        return True
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update a rule"""
        if rule_id not in self.rules:
            return False
        
        rule = self.rules[rule_id]
        
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        rule.updated_at = datetime.now()
        return True
    
    def add_ruleset(self, ruleset: RuleSet) -> bool:
        """Add a ruleset"""
        if ruleset.ruleset_id in self.rulesets:
            return False
        
        self.rulesets[ruleset.ruleset_id] = ruleset
        
        # Add all rules in the ruleset
        for rule in ruleset.rules:
            self.add_rule(rule)
        
        return True
    
    async def evaluate(self, data: Dict[str, Any],
                      rule_ids: Optional[List[str]] = None) -> List[RuleExecutionResult]:
        """Evaluate rules against data"""
        context = RuleExecutionContext(data=data)
        results = []
        
        # Get rules to evaluate
        if rule_ids:
            rules_to_evaluate = [self.rules[rid] for rid in rule_ids 
                                if rid in self.rules]
        else:
            rules_to_evaluate = list(self.rules.values())
        
        # Sort by priority
        rules_to_evaluate.sort(key=lambda r: r.priority, reverse=True)
        
        # Evaluate each rule
        for rule in rules_to_evaluate:
            if not rule.enabled:
                continue
            
            result = await self._evaluate_rule(rule, context)
            results.append(result)
            
            # Add to execution history
            self.execution_history.append(result)
            
            # Update metrics
            self._update_metrics(result)
        
        return results
    
    async def _evaluate_rule(self, rule: Rule,
                           context: RuleExecutionContext) -> RuleExecutionResult:
        """Evaluate a single rule"""
        start_time = datetime.now()
        
        result = RuleExecutionResult(
            rule_id=rule.rule_id,
            triggered=False,
            conditions_met=False,
            actions_executed=[],
            execution_time=0.0
        )
        
        try:
            # Check conditions
            if rule.conditions:
                conditions_met = self._evaluate_conditions(rule.conditions, context)
                result.conditions_met = conditions_met
                
                if not conditions_met:
                    result.execution_time = (datetime.now() - start_time).total_seconds()
                    return result
            else:
                result.conditions_met = True
            
            # Execute actions
            result.triggered = True
            rule.last_triggered = datetime.now()
            rule.trigger_count += 1
            
            for action in rule.actions:
                try:
                    action_result = await self._execute_action(action, context)
                    result.actions_executed.append(action.action_type.value)
                    result.results[action.action_type.value] = action_result
                except Exception as e:
                    result.errors.append(f"Action {action.action_type.value} failed: {str(e)}")
                    
                    if not action.retry_on_failure:
                        break
                    
                    # Retry logic
                    for retry in range(action.max_retries):
                        await asyncio.sleep(2 ** retry)  # Exponential backoff
                        try:
                            action_result = await self._execute_action(action, context)
                            result.actions_executed.append(action.action_type.value)
                            result.results[action.action_type.value] = action_result
                            break
                        except Exception:
                            continue
            
            # Add rule to execution history in context
            context.history.append(rule.rule_id)
            
        except Exception as e:
            result.errors.append(str(e))
        
        result.execution_time = (datetime.now() - start_time).total_seconds()
        return result
    
    def _evaluate_conditions(self, condition_group: ConditionGroup,
                           context: RuleExecutionContext) -> bool:
        """Evaluate a condition group"""
        if condition_group.operator == LogicalOperator.AND:
            return all(
                self._evaluate_single_condition(cond, context)
                if isinstance(cond, Condition)
                else self._evaluate_conditions(cond, context)
                for cond in condition_group.conditions
            )
        elif condition_group.operator == LogicalOperator.OR:
            return any(
                self._evaluate_single_condition(cond, context)
                if isinstance(cond, Condition)
                else self._evaluate_conditions(cond, context)
                for cond in condition_group.conditions
            )
        elif condition_group.operator == LogicalOperator.NOT:
            if condition_group.conditions:
                first_condition = condition_group.conditions[0]
                if isinstance(first_condition, Condition):
                    return not self._evaluate_single_condition(first_condition, context)
                else:
                    return not self._evaluate_conditions(first_condition, context)
            return True
        elif condition_group.operator == LogicalOperator.XOR:
            results = [
                self._evaluate_single_condition(cond, context)
                if isinstance(cond, Condition)
                else self._evaluate_conditions(cond, context)
                for cond in condition_group.conditions
            ]
            return sum(results) == 1
        
        return False
    
    def _evaluate_single_condition(self, condition: Condition,
                                  context: RuleExecutionContext) -> bool:
        """Evaluate a single condition"""
        # Get field value from context
        field_value = self._get_field_value(condition.field, context)
        
        # Convert types if needed
        if condition.data_type:
            field_value = self._convert_type(field_value, condition.data_type)
            condition_value = self._convert_type(condition.value, condition.data_type)
        else:
            condition_value = condition.value
        
        # Case sensitivity
        if not condition.case_sensitive and isinstance(field_value, str):
            field_value = field_value.lower()
            if isinstance(condition_value, str):
                condition_value = condition_value.lower()
        
        # Evaluate based on operator
        if condition.operator == ConditionOperator.EQUALS:
            return field_value == condition_value
        elif condition.operator == ConditionOperator.NOT_EQUALS:
            return field_value != condition_value
        elif condition.operator == ConditionOperator.GREATER_THAN:
            return field_value > condition_value
        elif condition.operator == ConditionOperator.GREATER_THAN_OR_EQUAL:
            return field_value >= condition_value
        elif condition.operator == ConditionOperator.LESS_THAN:
            return field_value < condition_value
        elif condition.operator == ConditionOperator.LESS_THAN_OR_EQUAL:
            return field_value <= condition_value
        elif condition.operator == ConditionOperator.CONTAINS:
            return condition_value in str(field_value)
        elif condition.operator == ConditionOperator.NOT_CONTAINS:
            return condition_value not in str(field_value)
        elif condition.operator == ConditionOperator.STARTS_WITH:
            return str(field_value).startswith(str(condition_value))
        elif condition.operator == ConditionOperator.ENDS_WITH:
            return str(field_value).endswith(str(condition_value))
        elif condition.operator == ConditionOperator.REGEX:
            return bool(re.match(str(condition_value), str(field_value)))
        elif condition.operator == ConditionOperator.IN:
            return field_value in condition_value
        elif condition.operator == ConditionOperator.NOT_IN:
            return field_value not in condition_value
        elif condition.operator == ConditionOperator.BETWEEN:
            if isinstance(condition_value, (list, tuple)) and len(condition_value) == 2:
                return condition_value[0] <= field_value <= condition_value[1]
            return False
        elif condition.operator == ConditionOperator.IS_NULL:
            return field_value is None
        elif condition.operator == ConditionOperator.IS_NOT_NULL:
            return field_value is not None
        elif condition.operator == ConditionOperator.IS_TRUE:
            return bool(field_value) is True
        elif condition.operator == ConditionOperator.IS_FALSE:
            return bool(field_value) is False
        
        return False
    
    def _get_field_value(self, field_path: str,
                        context: RuleExecutionContext) -> Any:
        """Get field value from context using dot notation"""
        # Check if it's a function call
        if field_path.startswith("@"):
            func_name = field_path[1:].split("(")[0]
            if func_name in self.custom_functions:
                # Extract arguments if any
                if "(" in field_path and ")" in field_path:
                    args_str = field_path[field_path.index("(") + 1:field_path.index(")")]
                    args = [arg.strip() for arg in args_str.split(",") if arg.strip()]
                    return self.custom_functions[func_name](context, *args)
                else:
                    return self.custom_functions[func_name](context)
        
        # Navigate nested structure
        parts = field_path.split(".")
        value = context.data
        
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list) and part.isdigit():
                index = int(part)
                if 0 <= index < len(value):
                    value = value[index]
                else:
                    return None
            else:
                # Try to get attribute
                try:
                    value = getattr(value, part)
                except (AttributeError, TypeError):
                    return None
        
        return value
    
    def _convert_type(self, value: Any, data_type: str) -> Any:
        """Convert value to specified type"""
        if value is None:
            return None
        
        try:
            if data_type == "string":
                return str(value)
            elif data_type == "number":
                return float(value)
            elif data_type == "integer":
                return int(value)
            elif data_type == "boolean":
                return bool(value)
            elif data_type == "date":
                if isinstance(value, str):
                    return datetime.fromisoformat(value)
                return value
            elif data_type == "json":
                if isinstance(value, str):
                    return json.loads(value)
                return value
        except (ValueError, TypeError):
            pass
        
        return value
    
    async def _execute_action(self, action: Action,
                            context: RuleExecutionContext) -> Any:
        """Execute an action"""
        if action.action_type not in self.action_handlers:
            raise ValueError(f"No handler for action type: {action.action_type}")
        
        handler = self.action_handlers[action.action_type]
        
        # Apply timeout if specified
        if action.timeout:
            return await asyncio.wait_for(
                handler(action, context),
                timeout=action.timeout
            )
        else:
            return await handler(action, context)
    
    def _update_metrics(self, result: RuleExecutionResult):
        """Update execution metrics"""
        self.metrics["total_executions"] += 1
        
        if result.triggered:
            if not result.errors:
                self.metrics["successful_executions"] += 1
            else:
                self.metrics["failed_executions"] += 1
        
        # Update average execution time
        total = self.metrics["total_executions"]
        current_avg = self.metrics["average_execution_time"]
        self.metrics["average_execution_time"] = (
            (current_avg * (total - 1) + result.execution_time) / total
        )
    
    def register_action_handler(self, action_type: ActionType,
                               handler: Callable):
        """Register an action handler"""
        self.action_handlers[action_type] = handler
    
    def register_custom_function(self, name: str, function: Callable):
        """Register a custom function for use in conditions"""
        self.custom_functions[name] = function
    
    async def process_event(self, event: Dict[str, Any]) -> List[RuleExecutionResult]:
        """Process an event through the rules engine"""
        # Find matching rules based on event type
        matching_rules = []
        
        for rule in self.rules.values():
            if rule.rule_type == RuleType.TRIGGER and rule.enabled:
                # Check if rule matches event pattern
                if self._matches_event(rule, event):
                    matching_rules.append(rule.rule_id)
        
        # Evaluate matching rules
        if matching_rules:
            return await self.evaluate(event, matching_rules)
        
        return []
    
    def _matches_event(self, rule: Rule, event: Dict[str, Any]) -> bool:
        """Check if rule matches event"""
        # Simple pattern matching based on metadata
        event_type = event.get("type")
        rule_event_type = rule.metadata.get("event_type")
        
        if rule_event_type and event_type:
            return rule_event_type == event_type
        
        return True  # Default to matching if no specific type
    
    # Default action handlers
    async def _handle_log_action(self, action: Action,
                                context: RuleExecutionContext) -> Dict[str, Any]:
        """Handle log action"""
        message = action.parameters.get("message", "Rule triggered")
        level = action.parameters.get("level", "info")
        
        # Format message with context variables
        formatted_message = self._format_string(message, context)
        
        print(f"[{level.upper()}] {formatted_message}")
        
        return {"logged": True, "message": formatted_message}
    
    async def _handle_transform_action(self, action: Action,
                                     context: RuleExecutionContext) -> Dict[str, Any]:
        """Handle transform action"""
        transformations = action.parameters.get("transformations", [])
        results = {}
        
        for transform in transformations:
            source_field = transform.get("source")
            target_field = transform.get("target")
            operation = transform.get("operation")
            
            source_value = self._get_field_value(source_field, context)
            
            if operation == "uppercase":
                results[target_field] = str(source_value).upper()
            elif operation == "lowercase":
                results[target_field] = str(source_value).lower()
            elif operation == "trim":
                results[target_field] = str(source_value).strip()
            elif operation == "replace":
                old = transform.get("old", "")
                new = transform.get("new", "")
                results[target_field] = str(source_value).replace(old, new)
            elif operation == "concat":
                values = [self._get_field_value(f, context) 
                         for f in transform.get("fields", [])]
                separator = transform.get("separator", "")
                results[target_field] = separator.join(map(str, values))
            elif operation == "split":
                separator = transform.get("separator", ",")
                results[target_field] = str(source_value).split(separator)
            elif operation == "calculate":
                expression = transform.get("expression")
                results[target_field] = self._evaluate_expression(expression, context)
            else:
                results[target_field] = source_value
        
        # Update context with transformed data
        context.data.update(results)
        
        return results
    
    async def _handle_filter_action(self, action: Action,
                                  context: RuleExecutionContext) -> Dict[str, Any]:
        """Handle filter action"""
        filter_conditions = action.parameters.get("conditions", [])
        passed = True
        
        for condition_dict in filter_conditions:
            condition = Condition(
                field=condition_dict.get("field"),
                operator=ConditionOperator(condition_dict.get("operator")),
                value=condition_dict.get("value")
            )
            
            if not self._evaluate_single_condition(condition, context):
                passed = False
                break
        
        return {"passed": passed}
    
    async def _handle_notify_action(self, action: Action,
                                  context: RuleExecutionContext) -> Dict[str, Any]:
        """Handle notify action"""
        notification_type = action.parameters.get("type", "log")
        message = action.parameters.get("message", "Notification")
        
        formatted_message = self._format_string(message, context)
        
        # Placeholder for actual notification
        print(f"[NOTIFICATION] {notification_type}: {formatted_message}")
        
        return {"notified": True, "message": formatted_message}
    
    async def _handle_store_action(self, action: Action,
                                 context: RuleExecutionContext) -> Dict[str, Any]:
        """Handle store action"""
        storage_type = action.parameters.get("type", "context")
        key = action.parameters.get("key")
        value = action.parameters.get("value")
        
        if storage_type == "context":
            context.variables[key] = value
        elif storage_type == "result":
            context.results[key] = value
        
        return {"stored": True, "key": key}
    
    async def _handle_delay_action(self, action: Action,
                                 context: RuleExecutionContext) -> Dict[str, Any]:
        """Handle delay action"""
        seconds = action.parameters.get("seconds", 1)
        await asyncio.sleep(seconds)
        return {"delayed": seconds}
    
    async def _handle_webhook_action(self, action: Action,
                                   context: RuleExecutionContext) -> Dict[str, Any]:
        """Handle webhook action"""
        # Placeholder for webhook implementation
        url = action.parameters.get("url")
        method = action.parameters.get("method", "POST")
        headers = action.parameters.get("headers", {})
        body = action.parameters.get("body", {})
        
        # Format body with context data
        formatted_body = self._format_dict(body, context)
        
        return {
            "webhook_sent": True,
            "url": url,
            "method": method,
            "body": formatted_body
        }
    
    def _format_string(self, template: str,
                      context: RuleExecutionContext) -> str:
        """Format string with context variables"""
        # Replace {{field}} with actual values
        import re
        
        def replacer(match):
            field = match.group(1)
            value = self._get_field_value(field, context)
            return str(value) if value is not None else ""
        
        return re.sub(r'\{\{([^}]+)\}\}', replacer, template)
    
    def _format_dict(self, template: Dict[str, Any],
                    context: RuleExecutionContext) -> Dict[str, Any]:
        """Format dictionary with context variables"""
        result = {}
        
        for key, value in template.items():
            if isinstance(value, str):
                result[key] = self._format_string(value, context)
            elif isinstance(value, dict):
                result[key] = self._format_dict(value, context)
            elif isinstance(value, list):
                result[key] = [
                    self._format_string(item, context) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        
        return result
    
    def _evaluate_expression(self, expression: str,
                           context: RuleExecutionContext) -> Any:
        """Evaluate a mathematical or logical expression"""
        # Simple expression evaluator
        # In production, use a proper expression parser
        try:
            # Replace field references with values
            import re
            
            def replacer(match):
                field = match.group(1)
                value = self._get_field_value(field, context)
                return str(value) if value is not None else "0"
            
            evaluated = re.sub(r'\{\{([^}]+)\}\}', replacer, expression)
            
            # Safe evaluation of simple expressions
            allowed_names = {
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "len": len,
                "round": round
            }
            
            return eval(evaluated, {"__builtins__": {}}, allowed_names)
            
        except Exception:
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get engine metrics"""
        return self.metrics
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get rule statistics"""
        stats = {
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for r in self.rules.values() if r.enabled),
            "total_rulesets": len(self.rulesets),
            "rules_by_type": {},
            "most_triggered": []
        }
        
        # Count by type
        for rule in self.rules.values():
            rule_type = rule.rule_type.value
            stats["rules_by_type"][rule_type] = stats["rules_by_type"].get(rule_type, 0) + 1
        
        # Find most triggered rules
        triggered_rules = sorted(
            [(r.trigger_count, r.name) for r in self.rules.values() if r.trigger_count > 0],
            reverse=True
        )
        stats["most_triggered"] = triggered_rules[:10]
        
        return stats