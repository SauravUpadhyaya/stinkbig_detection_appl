import io
import os
import sqlite3
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
import re 

import numpy as np
import pandas as pd
import smtplib
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from streamlit_autorefresh import st_autorefresh
from ultralytics import YOLO
import plotly.io as pio
from sam2_verification import get_sam2_verifier
import ollama
import plotly.express as px

BASE_DIR = Path.cwd()
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)



DB_PATH = BASE_DIR / "stinkbug.db"
MODEL_PATH = BASE_DIR / "yolov8m_cbam_asff_finetuned.pt"
THRESHOLD_PER_100 = 16
MIN_IMAGES_FOR_ALERT = 100
MAX_GALLERY_ITEMS = 20


import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum




import os
import torch
import streamlit as st

def download_weights():
    # Path for SAM 2
    if not os.path.exists("checkpoints/sam2_hiera_small.pt"):
        os.makedirs("checkpoints", exist_ok=True)
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
        torch.hub.download_url_to_file(url, "checkpoints/sam2_hiera_small.pt")

    # # Path for GroundingDINO
    # if not os.path.exists("checkpoints/groundingdino_swint_ogc.pth"):
    #     url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    #     torch.hub.download_url_to_file(url, "checkpoints/groundingdino_swint_ogc.pth")

download_weights()


class ToolStatus(Enum):
    """Status of tool execution"""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"


@dataclass
class ToolResult:
    """Result from tool execution"""
    tool_name: str
    status: ToolStatus
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class AgentStep:
    """Single step in agent's reasoning chain"""
    step_num: int
    thought: str
    action: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None
    observation: Optional[str] = None
    result: Optional[ToolResult] = None


@dataclass
class AgentPlan:
    """Multi-step plan created by agent"""
    goal: str
    steps: List[str] = field(default_factory=list)
    dependencies: Dict[int, List[int]] = field(default_factory=dict)
    completed_steps: List[int] = field(default_factory=list)


class AgentTools:
    """Collection of tools the agent can use"""
    
    @staticmethod
    def query_location_stats(location: str) -> Dict:
        """Get statistics for a specific location"""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) as image_count, SUM(insect_count) as total_insects FROM reports WHERE location = ?",
                    (location,)
                )
                row = cursor.fetchone()
                image_count, total_insects = row[0], row[1] or 0
                density = (total_insects / image_count * 100) if image_count > 0 else 0
                
                return {
                    "location": location,
                    "image_count": image_count,
                    "total_insects": total_insects,
                    "density_per_100": round(density, 2),
                    "threshold_exceeded": density > THRESHOLD_PER_100
                }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def list_all_locations() -> List[str]:
        """Get list of all locations in database"""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT location FROM reports ORDER BY location")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            return []
    
    @staticmethod
    def search_reports(location: Optional[str] = None, min_insects: Optional[int] = None, 
                      date_from: Optional[str] = None) -> List[Dict]:
        """Search reports with filters"""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                query = "SELECT id, username, report_date, location, image_name, insect_count FROM reports WHERE 1=1"
                params = []
                
                if location:
                    query += " AND location = ?"
                    params.append(location)
                if min_insects:
                    query += " AND insect_count >= ?"
                    params.append(min_insects)
                if date_from:
                    query += " AND report_date >= ?"
                    params.append(date_from)
                
                query += " ORDER BY report_date DESC LIMIT 20"
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        "id": row[0],
                        "username": row[1],
                        "date": row[2],
                        "location": row[3],
                        "image_name": row[4],
                        "insect_count": row[5]
                    })
                return results
        except Exception as e:
            return [{"error": str(e)}]
    
    @staticmethod
    def get_images_from_location(location: str, limit: int = 5) -> List[Dict]:
        """Get images from a specific location"""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT image_name, image_bytes, created_at FROM gallery WHERE location = ? ORDER BY created_at DESC LIMIT ?",
                    (location, limit)
                )
                results = []
                for row in cursor.fetchall():
                    results.append({
                        "image_name": row[0],
                        "has_image": bool(row[1]),
                        "created_at": row[2]
                    })
                return results
        except Exception as e:
            return [{"error": str(e)}]
    
    @staticmethod
    def compare_locations(locations: List[str]) -> Dict:
        """Compare statistics across multiple locations"""
        results = {}
        for loc in locations[:5]:  # Limit to 5 locations
            results[loc] = AgentTools.query_location_stats(loc)
        
        # Add comparison summary
        if results:
            densities = [(loc, data.get('density_per_100', 0)) for loc, data in results.items() if 'error' not in data]
            if densities:
                highest = max(densities, key=lambda x: x[1])
                lowest = min(densities, key=lambda x: x[1])
                results['_summary'] = {
                    "highest_density": {"location": highest[0], "density": highest[1]},
                    "lowest_density": {"location": lowest[0], "density": lowest[1]},
                    "average_density": round(sum(d[1] for d in densities) / len(densities), 2)
                }
        
        return results
    
    @staticmethod
    def search_feedback(query: str, limit: int = 5) -> List[Dict]:
        """Search thumbs-up feedback for similar questions"""
        try:
            query_words = set(query.lower().split())
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT user_query, assistant_response, response_type, created_at FROM feedback WHERE label = 'yes' ORDER BY created_at DESC LIMIT 50"
                )
                
                matches = []
                for row in cursor.fetchall():
                    stored_query = row[0]
                    stored_words = set(stored_query.lower().split())
                    if stored_words:
                        similarity = len(query_words & stored_words) / len(query_words | stored_words)
                        if similarity > 0.3:
                            matches.append({
                                "query": stored_query,
                                "answer": row[1],
                                "type": row[2],
                                "date": row[3],
                                "similarity": round(similarity, 3)
                            })
                
                matches.sort(key=lambda x: x['similarity'], reverse=True)
                return matches[:limit]
        except Exception as e:
            return [{"error": str(e)}]
    
    @staticmethod
    def calculate_trend(location: str, period_days: int = 30) -> Dict:
        """Calculate detailed trends for a location over time with week-by-week breakdown"""
        try:
            from datetime import datetime, timedelta
            
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                # Get all historical data for this location
                cursor.execute(
                    "SELECT report_date, insect_count FROM reports WHERE location = ? ORDER BY report_date",
                    (location,)
                )
                
                data = cursor.fetchall()
                if not data:
                    return {"error": f"No data for {location}"}
                
                # Parse dates and counts (handle malformed dates)
                reports = []
                for report_date, insect_count in data:
                    try:
                        # Try ISO format first
                        date_obj = datetime.fromisoformat(report_date)
                    except:
                        # If date is malformed, just use current date as placeholder
                        # This ensures we still get statistics even if dates are bad
                        date_obj = datetime.now()
                    
                    reports.append({
                        'date': date_obj,
                        'count': insect_count or 0,
                        'raw_date': report_date
                    })
                
                if not reports:
                    return {"error": "No valid count data"}
                
                # Sort by date (even if dates are malformed)
                reports.sort(key=lambda x: x['date'])
                
                # Calculate overall statistics
                all_counts = [r['count'] for r in reports]
                overall_avg = sum(all_counts) / len(all_counts) if all_counts else 0
                max_count = max(all_counts) if all_counts else 0
                min_count = min(all_counts) if all_counts else 0
                total_insects = sum(all_counts)
                
                # If dates are malformed (all same), just provide aggregate stats
                unique_dates = len(set(r['raw_date'] for r in reports))
                
                if unique_dates == 1:
                    # All records have same date - can't do week-by-week
                    return {
                        "location": location,
                        "total_reports": len(data),
                        "overall_average": round(overall_avg, 2),
                        "total_insects": total_insects,
                        "min_insects": min_count,
                        "max_insects": max_count,
                        "trend": "  Insufficient Date Data",
                        "trend_description": "All records have the same timestamp. Cannot calculate trends over time.",
                        "change_percent": 0,
                        "week_by_week": [{
                            'week': 'All Records',
                            'reports': len(data),
                            'total_insects': total_insects,
                            'average': round(overall_avg, 2)
                        }],
                        "summary": f"{location}: {overall_avg:.1f} insects per report (average)"
                    }
                
                # Week-by-week breakdown for valid dates
                weeks = {}
                start_date = reports[0]['date']
                
                for report in reports:
                    days_from_start = (report['date'] - start_date).days
                    # Group into weeks, but allow single-day reports if only 1 day
                    if days_from_start == 0:
                        week_key = "week_1"
                    else:
                        week_num = (days_from_start // 7) + 1
                        week_key = f"week_{week_num}"
                    
                    if week_key not in weeks:
                        weeks[week_key] = {
                            'count': 0,
                            'total_insects': 0,
                            'reports': []
                        }
                    
                    weeks[week_key]['count'] += 1
                    weeks[week_key]['total_insects'] += report['count']
                    weeks[week_key]['reports'].append(report['date'].strftime("%Y-%m-%d"))
                
                # Calculate week averages and trend
                week_data = []
                for week_key in sorted(weeks.keys()):
                    week = weeks[week_key]
                    avg_insects = week['total_insects'] / week['count'] if week['count'] > 0 else 0
                    week_data.append({
                        'week': week_key.replace('week_', 'Week '),
                        'reports': week['count'],
                        'total_insects': week['total_insects'],
                        'average': round(avg_insects, 2)
                    })
                
                # Detect trend direction
                if len(week_data) >= 2:
                    first_half_avg = sum(w['average'] for w in week_data[:len(week_data)//2]) / max(1, len(week_data)//2)
                    second_half_avg = sum(w['average'] for w in week_data[len(week_data)//2:]) / max(1, len(week_data) - len(week_data)//2)
                    
                    change_percent = ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg > 0 else 0
                    
                    if change_percent > 10:
                        trend = "📈 Increasing"
                        trend_desc = f"Insect counts rising ({change_percent:.1f}% increase)"
                    elif change_percent < -10:
                        trend = "📉 Decreasing"
                        trend_desc = f"Insect counts falling ({abs(change_percent):.1f}% decrease)"
                    else:
                        trend = "➡️ Stable"
                        trend_desc = f"Relatively stable ({abs(change_percent):.1f}% change)"
                else:
                    trend = "➡️ Stable"
                    trend_desc = "Limited data for trend detection"
                    change_percent = 0
                
                return {
                    "location": location,
                    "total_reports": len(data),
                    "overall_average": round(overall_avg, 2),
                    "total_insects": total_insects,
                    "min_insects": min_count,
                    "max_insects": max_count,
                    "trend": trend,
                    "trend_description": trend_desc,
                    "change_percent": round(change_percent, 1),
                    "week_by_week": week_data,
                    "first_date": reports[0]['raw_date'],
                    "last_date": reports[-1]['raw_date'],
                    "summary": f"{location}: {trend} - {trend_desc}"
                }
        except Exception as e:
            return {"error": f"Trend analysis failed: {str(e)}"}


class AdvancedAgent:
    """Advanced agentic LLM with planning, tool use, and self-correction"""
    
    def __init__(self, ollama_client, model: str = "mistral", verbose: bool = True):
        self.client = ollama_client
        self.model = model
        self.verbose = verbose
        self.tools = self._register_tools()
        self.execution_history: List[AgentStep] = []
        self.max_iterations = 10
        
    def _register_tools(self) -> Dict[str, Callable]:
        """Register available tools"""
        return {
            "query_location_stats": AgentTools.query_location_stats,
            "list_all_locations": AgentTools.list_all_locations,
            "search_reports": AgentTools.search_reports,
            "get_images_from_location": AgentTools.get_images_from_location,
            "compare_locations": AgentTools.compare_locations,
            "search_feedback": AgentTools.search_feedback,
            "calculate_trend": AgentTools.calculate_trend,
        }
    
    def _get_tool_descriptions(self) -> str:
        """Generate tool descriptions for LLM"""
        descriptions = []
        descriptions.append("query_location_stats(location: str) -> Get stats for a location (image count, insects, density)")
        descriptions.append("list_all_locations() -> Get list of all available locations")
        descriptions.append("search_reports(location?, min_insects?, date_from?) -> Search reports with filters")
        descriptions.append("get_images_from_location(location: str, limit: int) -> Get images from location")
        descriptions.append("compare_locations(locations: List[str]) -> Compare multiple locations")
        descriptions.append("search_feedback(query: str, limit: int) -> Find similar answered questions")
        descriptions.append("calculate_trend(location: str, period_days: int) -> Calculate trends over time")
        return "\n".join(descriptions)
    
    def create_plan(self, user_query: str) -> AgentPlan:
        """Create multi-step plan for complex queries"""
        planning_prompt = f"""You are a planning agent. Break down this query into clear steps.

User Query: {user_query}

Available Tools:
{self._get_tool_descriptions()}

Create a numbered plan with 2-5 steps. Each step should be one clear action.
Format:
1. [action description]
2. [action description]
...

Plan:"""
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': planning_prompt}],
                options={"temperature": 0.2, "num_predict": 300}
            )
            
            plan_text = response['message']['content']
            steps = []
            for line in plan_text.split('\n'):
                line = line.strip()
                if line and line[0].isdigit():
                    steps.append(line.split('.', 1)[1].strip() if '.' in line else line)
            
            return AgentPlan(goal=user_query, steps=steps)
        except Exception as e:
            if self.verbose:
                print(f"Planning failed: {e}")
            return AgentPlan(goal=user_query, steps=[user_query])
    
    def _make_json_safe(self, obj) -> any:
        """Convert non-JSON-serializable objects to safe representations"""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, bytes):
            return f"<binary data: {len(obj)} bytes>"
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def execute_tool(self, tool_name: str, args: Dict) -> ToolResult:
        """Execute a single tool"""
        start_time = time.time()
        
        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name,
                status=ToolStatus.FAILED,
                result=None,
                error=f"Tool '{tool_name}' not found",
                execution_time=time.time() - start_time
            )
        
        try:
            result = self.tools[tool_name](**args)
            return ToolResult(
                tool_name=tool_name,
                status=ToolStatus.SUCCESS,
                result=result,
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                status=ToolStatus.FAILED,
                result=None,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def execute_parallel_tools(self, tool_calls: List[Tuple[str, Dict]]) -> List[ToolResult]:
        """Execute multiple tools in parallel"""
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.execute_tool, tool_name, args): (tool_name, args)
                for tool_name, args in tool_calls
            }
            
            results = []
            for future in as_completed(futures):
                results.append(future.result())
            
            return results
    
    def extract_tool_call(self, thought: str) -> Optional[Tuple[str, Dict]]:
        """Extract tool call from LLM thought"""
        # Look for patterns like: use query_location_stats with location="Council Bluffs"
        import re
        
        # Match: tool_name(arg1=value1, arg2=value2)
        pattern = r'(\w+)\((.*?)\)'
        match = re.search(pattern, thought)
        
        if not match:
            # Try alternative pattern: use tool_name with arg=value
            pattern2 = r'(?:use|call)\s+(\w+)\s+with\s+(.*?)(?:\.|$)'
            match = re.search(pattern2, thought, re.IGNORECASE)
            
            if match:
                tool_name = match.group(1)
                args_str = match.group(2)
                args = {}
                
                # Parse key=value pairs
                for arg_match in re.finditer(r'(\w+)\s*=\s*["\']?([^,"\'\s]+)["\']?', args_str):
                    key, value = arg_match.groups()
                    # Try to convert to appropriate type
                    try:
                        args[key] = int(value)
                    except:
                        try:
                            args[key] = float(value)
                        except:
                            args[key] = value
                
                return (tool_name, args) if tool_name in self.tools else None
        else:
            tool_name = match.group(1)
            if tool_name not in self.tools:
                return None
            
            args_str = match.group(2)
            args = {}
            
            if args_str.strip():
                # Parse arguments
                for arg_pair in args_str.split(','):
                    if '=' in arg_pair:
                        key, value = arg_pair.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        try:
                            args[key] = json.loads(value)
                        except:
                            args[key] = value
            
            return (tool_name, args)
        
        return None
    
    def self_correct(self, step: AgentStep, user_query: str) -> Optional[str]:
        """Validate and correct agent's reasoning"""
        if not step.result or step.result.status == ToolStatus.FAILED:
            correction_prompt = f"""The previous action failed.

Original Query: {user_query}
Failed Action: {step.action}
Error: {step.result.error if step.result else 'Unknown'}

What should we do instead? Provide ONE alternative action.
Alternative:"""
            
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=[{'role': 'user', 'content': correction_prompt}],
                    options={"temperature": 0.3, "num_predict": 150}
                )
                return response['message']['content'].strip()
            except:
                return None
        
        return None
    
    def run(self, user_query: str, context: Dict = None) -> Dict:
        """Main agent execution loop with chain-of-thought, robust self-correction, and parallel execution."""
        if self.verbose:
            print(f"\n🤖 Agent starting on: {user_query}")

        query_lower = user_query.lower()
        requires_reasoning = any(word in query_lower for word in [
            "why", "reason", "explain", "understand", "effect", "impact", "consequence",
            "cause", "help me", "what could be"
        ])

        # Try direct data formatting first ONLY if no reasoning required
        if not requires_reasoning:
            direct_answer = self.format_answer_from_data(user_query)
            if direct_answer:
                if self.verbose:
                    print("✅ Using direct database answer (validated)")
                return direct_answer
        else:
            if self.verbose:
                print("🧠 Reasoning required - executing tools directly for synthesis")
            locations = AgentTools.list_all_locations()
            if locations:
                comparison_result = AgentTools.compare_locations(locations)
                self.execution_history = [
                    AgentStep(
                        step_num=1,
                        thought="User wants reasoning about highest/lowest stinkbug populations",
                        action="compare_locations",
                        tool_name="compare_locations",
                        observation=json.dumps(self._make_json_safe(comparison_result), indent=2),
                        result=ToolResult(
                            tool_name="compare_locations",
                            status=ToolStatus.SUCCESS,
                            result=comparison_result,
                            execution_time=0
                        )
                    )
                ]
                return self.synthesize_answer(user_query)

        # Create plan
        plan = self.create_plan(user_query)
        if self.verbose:
            print(f"📋 Plan created with {len(plan.steps)} steps")

        self.execution_history = []
        conversation_history = []

        # Add context if provided
        context_str = ""
        if context:
            try:
                safe_context = self._make_json_safe(context)
                context_str = f"\n\nContext:\n{json.dumps(safe_context, indent=2)}"
            except:
                context_str = f"\n\nContext:\n{str(context)}"

        system_prompt = f"""You are an expert AI agent for stinkbug detection and pest management analysis.

Available Tools:
{self._get_tool_descriptions()}

{context_str}

Use chain-of-thought reasoning. For each step:
1. Think about what you need to do
2. Choose a tool and specify arguments clearly
3. Wait for the result
4. Reason about the result

Format your response as:
Thought: [your reasoning]
Action: [tool_name with arguments]"""

        parallel_candidates = []
        sequential_steps = []
        used_tools = set()
        for plan_step in plan.steps:
            tool_call = self.extract_tool_call(plan_step)
            if tool_call and tool_call[0] not in used_tools:
                parallel_candidates.append(tool_call)
                used_tools.add(tool_call[0])
            else:
                sequential_steps.append(plan_step)

        if len(parallel_candidates) > 1:
            if self.verbose:
                print(f"⚡ Running {len(parallel_candidates)} tool calls in parallel")
            parallel_results = self.execute_parallel_tools(parallel_candidates)
            for idx, (tool_call, result) in enumerate(zip(parallel_candidates, parallel_results), 1):
                tool_name, args = tool_call
                safe_result = self._make_json_safe(result.result)
                observation = json.dumps(safe_result, indent=2) if result.status == ToolStatus.SUCCESS else f"Error: {result.error}"
                step = AgentStep(
                    step_num=idx,
                    thought=f"Parallel execution of {tool_name}",
                    action=f"{tool_name}({args})",
                    tool_name=tool_name,
                    tool_args=args,
                    observation=observation,
                    result=result
                )
                self.execution_history.append(step)
            # Continue with sequential steps if any
            plan_steps_to_run = sequential_steps
        else:
            plan_steps_to_run = plan.steps

        # --- Improved: Sequential execution with robust self-correction and retry loop ---
        for step_num, plan_step in enumerate(plan_steps_to_run, 1):
            if step_num > self.max_iterations:
                break
            history_text = "\n".join([
                f"Step {s.step_num}: {s.thought}\nAction: {s.action}\nObservation: {s.observation}"
                for s in self.execution_history
            ])
            reasoning_prompt = f"""{system_prompt}

Original Query: {user_query}
Current Step: {plan_step}

Previous steps:
{history_text if history_text else "None yet"}

Your turn:"""
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=[{'role': 'user', 'content': reasoning_prompt}],
                    options={"temperature": 0.2, "num_predict": 400}
                )
                agent_response = response['message']['content']
                thought = ""
                action = ""
                lines = agent_response.split('\n')
                for line in lines:
                    if line.startswith('Thought:'):
                        thought = line.replace('Thought:', '').strip()
                    elif line.startswith('Action:'):
                        action = line.replace('Action:', '').strip()
                if not thought or not action:
                    thought = agent_response[:200]
                    action = plan_step
                tool_call = self.extract_tool_call(action)
                retry_count = 0
                max_retries = 3
                result = None
                observation = ""
                while retry_count < max_retries:
                    if tool_call:
                        tool_name, args = tool_call
                        result = self.execute_tool(tool_name, args)
                        safe_result = None
                        try:
                            safe_result = self._make_json_safe(result.result)
                            observation = json.dumps(safe_result, indent=2) if result.status == ToolStatus.SUCCESS else f"Error: {result.error}"
                        except Exception as je:
                            if result.status == ToolStatus.SUCCESS:
                                try:
                                    observation = str(result.result) if safe_result is None else str(safe_result)
                                except:
                                    observation = f"[Tool returned complex data for {tool_name}]"
                            else:
                                observation = f"Error: {result.error}"
                        step = AgentStep(
                            step_num=step_num,
                            thought=thought,
                            action=action,
                            tool_name=tool_name,
                            tool_args=args,
                            observation=observation,
                            result=result
                        )
                        if result.status == ToolStatus.SUCCESS:
                            break
                        else:
                            correction = self.self_correct(step, user_query)
                            if correction and self.verbose:
                                print(f"🔧 Self-correcting: {correction}")
                            # Try to extract new tool call from correction
                            tool_call = self.extract_tool_call(correction)
                            retry_count += 1
                    else:
                        step = AgentStep(
                            step_num=step_num,
                            thought=thought,
                            action=action,
                            observation="No tool executed"
                        )
                        break
                self.execution_history.append(step)
                if self.verbose:
                    print(f"\nStep {step_num}:")
                    print(f"  Thought: {thought[:100]}...")
                    print(f"  Action: {action[:100]}...")
                    if step.result:
                        print(f"  Result: {step.result.status.value} ({step.result.execution_time:.2f}s)")
            except Exception as e:
                if self.verbose:
                    print(f"Step {step_num} error: {e}")
                break
        return self.synthesize_answer(user_query)
    
    def format_answer_from_data(self, user_query: str) -> Optional[Dict]:
        """Format answer directly from database results without LLM synthesis.
        Returns structured answer for common query patterns."""
        
        query_lower = user_query.lower()
        
        # Pattern: "which city/location has lowest/highest..."
        if any(word in query_lower for word in ["which", "what"]) and any(word in query_lower for word in ["city", "location", "place"]):
            
            # Get all locations and their stats
            try:
                locations = AgentTools.list_all_locations()
                if not locations:
                    return None
                
                all_stats = []
                for loc in locations:
                    stats = AgentTools.query_location_stats(loc)
                    if 'error' not in stats:
                        all_stats.append({
                            'location': loc,
                            'total_insects': stats.get('total_insects', 0),
                            'image_count': stats.get('image_count', 0),
                            'density': stats.get('density_per_100', 0)
                        })
                
                if not all_stats:
                    return None
                
                # Determine what user is asking for
                if any(word in query_lower for word in ["lowest", "least", "fewest", "minimum", "smallest"]):
                    # Find lowest
                    if "density" in query_lower:
                        result = min(all_stats, key=lambda x: x['density'])
                        answer = f"**{result['location']}** has the lowest density with **{result['density']} insects per 100 images** ({result['total_insects']} total RBSB across {result['image_count']} image(s))."
                    else:
                        result = min(all_stats, key=lambda x: x['total_insects'])
                        answer = f"**{result['location']}** has the lowest number of stink bug(s) with **{result['total_insects']} total insects** across {result['image_count']} image(s)."
                    
                    # Add comparison context
                    all_stats_sorted = sorted(all_stats, key=lambda x: x['total_insects'])
                    answer += f"\n\n**All locations (sorted by count):**\n"
                    for i, stat in enumerate(all_stats_sorted, 1):
                        answer += f"{i}. {stat['location']}: {stat['total_insects']} insects ({stat['density']} per 100 images)\n"
                    
                    return {
                        "answer": answer,
                        "data_source": "direct_database_query",
                        "validated": True,
                        "raw_data": all_stats,
                        "success": True
                    }
                    
                elif any(word in query_lower for word in ["highest", "most", "maximum", "largest", "greatest"]):
                    # Find highest
                    if "density" in query_lower:
                        result = max(all_stats, key=lambda x: x['density'])
                        answer = f"**{result['location']}** has the highest density with **{result['density']} insects per 100 images** ({result['total_insects']} total insects across {result['image_count']} images)."
                    else:
                        result = max(all_stats, key=lambda x: x['total_insects'])
                        answer = f"**{result['location']}** has the highest number of stink bugs with **{result['total_insects']} total insects** across {result['image_count']} images."
                    
                    # Add comparison context
                    all_stats_sorted = sorted(all_stats, key=lambda x: x['total_insects'], reverse=True)
                    answer += f"\n\n**All locations (sorted by count):**\n"
                    for i, stat in enumerate(all_stats_sorted, 1):
                        answer += f"{i}. {stat['location']}: {stat['total_insects']} insects ({stat['density']} per 100 images)\n"
                    
                    return {
                        "answer": answer,
                        "data_source": "direct_database_query",
                        "validated": True,
                        "raw_data": all_stats,
                        "success": True
                    }
            except Exception as e:
                if self.verbose:
                    print(f"Direct data formatting error: {e}")
                return None
        
        return None
    
    def synthesize_answer(self, user_query: str) -> Dict:
        """Synthesize final answer from execution history, with explicit handling for missing comparison data."""
        observations = "\n\n".join([
            f"Step {s.step_num}: {s.thought}\nResult: {s.observation}"
            for s in self.execution_history
        ])

        has_trend_data = any("week_by_week" in str(s.observation) for s in self.execution_history)
        if has_trend_data:
            synthesis_prompt = f"""Based on the following trend analysis, provide a clear, detailed answer about the insect trends.

User Query: {user_query}

Analysis:
{observations}

Format your response with:
1. **Main Insight** - The primary trend (increasing/decreasing/stable)
2. **Week-by-Week Breakdown** - Show progression across weeks with emoji indicators
3. **Key Metrics** - Average, min, max insect counts
4. **Recommendation** - Any actionable insights based on the trend

Make the response detailed and data-driven.

Answer:"""
        else:
            # Check if user is asking for reasoning/analysis
            needs_reasoning = any(word in query_lower for word in [
                "why", "reason", "explain", "understand", "effect", "impact", "consequence",
                "cause", "help me", "what could be"
            ])
            print(f"\n=== AGENT SYNTHESIS ===")
            print(f"User query: {user_query[:80]}...")
            print(f"needs_reasoning: {needs_reasoning}")
            if needs_reasoning:
                synthesis_prompt = f"""You are analyzing stink bug detection data. The user needs both FACTS and REASONING.

User Query: {user_query}

Data from Tools:
{observations}

INSTRUCTIONS:
1. First, answer the factual parts (which location has highest/lowest stinkbugs) using the data above
2. Then explain WHY using your knowledge about stink bugs and environmental factors:
   - Climate and weather (warmer = more bugs)
   - Agriculture and crops (more crops = more bugs)
   - Habitat and vegetation (more hiding places = more bugs)
3. Then explain EFFECTS/CONSEQUENCES:
   - Crop damage and economic losses
   - Pest management challenges
   - Agricultural impact

Start with the facts, then explain the "why" and "what could be the effect".

Answer:"""
            else:
                synthesis_prompt = f"""Based on the following analysis, provide a clear, concise answer to the user.

User Query: {user_query}

Analysis:
{observations}

Provide a natural, helpful response that directly answers the user's question. Include specific data points and insights.

Answer:"""
        try:
            print(f"\n=== AGENT CALLING OLLAMA FOR SYNTHESIS ===")
            print(f"Model: {self.model}")
            print(f"Client: {self.client}")
            response = self.client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': synthesis_prompt}],
                options={"temperature": 0.3, "num_predict": 500}
            )
            print(f"Ollama synthesis successful")
            return {
                "answer": response['message']['content'],
                "steps_executed": len(self.execution_history),
                "tools_used": list(set([s.tool_name for s in self.execution_history if s.tool_name])),
                "execution_history": [
                    {
                        "step": s.step_num,
                        "thought": s.thought,
                        "action": s.action,
                        "tool": s.tool_name,
                        "status": s.result.status.value if s.result else "no_tool"
                    }
                    for s in self.execution_history
                ],
                "success": True
            }
        except Exception as e:
            print(f"\nAGENT SYNTHESIS ERROR: {e}")
            print(f"Error type: {type(e).__name__}")
            error_msg = str(e).lower()
            if "404" in error_msg or "not found" in error_msg:
                fallback = "📊 **Trend Analysis Results**\n\n"
                trend_found = False
                for step in self.execution_history:
                    if step.tool_name == "calculate_trend":
                        trend_found = True
                        obs = step.observation
                        try:
                            import json as json_module
                            trend_data = json_module.loads(obs) if isinstance(obs, str) else obs
                            if isinstance(trend_data, dict):
                                location = trend_data.get("location", "Unknown")
                                trend = trend_data.get("trend", "")
                                summary = trend_data.get("summary", "")
                                fallback += f"**Location:** {location}\n"
                                fallback += f"**Trend:** {trend}\n"
                                fallback += f"**Summary:** {summary}\n\n"
                                if "overall_average" in trend_data:
                                    fallback += "**Key Metrics:**\n"
                                    fallback += f"- Average insects per report: {trend_data.get('overall_average', 'N/A')}\n"
                                    fallback += f"- Total reports: {trend_data.get('total_reports', 'N/A')}\n"
                                    fallback += f"- Total insects: {trend_data.get('total_insects', 'N/A')}\n"
                                    fallback += f"- Min: {trend_data.get('min_insects', 'N/A')}\n"
                                    fallback += f"- Max: {trend_data.get('max_insects', 'N/A')}\n\n"
                                if "week_by_week" in trend_data:
                                    fallback += "**Week-by-Week Breakdown:**\n"
                                    for week in trend_data["week_by_week"]:
                                        fallback += f"- {week.get('week')}: {week.get('average')} avg insects ({week.get('reports')} reports)\n"
                                    fallback += "\n"
                        except:
                            fallback += f"**Raw Data:**\n{obs}\n\n"
                if not trend_found:
                    for step in self.execution_history:
                        if step.observation:
                            fallback += f"**{step.tool_name or 'Step ' + str(step.step_num)}:**\n{step.observation}\n\n"
                return {
                    "answer": fallback + "  *(Generated from tool data - Ollama LLM unavailable)*",
                    "steps_executed": len(self.execution_history),
                    "tools_used": list(set([s.tool_name for s in self.execution_history if s.tool_name])),
                    "success": False,
                    "error": "Ollama model not found. Please start Ollama: `ollama serve` and install mistral: `ollama pull mistral`"
                }
            return {
                "answer": f"I encountered an error synthesizing the answer: {e}\n\nPlease ensure Ollama is running and the 'mistral' model is available.",
                "steps_executed": len(self.execution_history),
                "success": False,
                "error": str(e)
            }

def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                report_date TEXT NOT NULL,
                location TEXT NOT NULL,
                image_name TEXT NOT NULL,
                insect_count INTEGER NOT NULL,
                priority TEXT DEFAULT 'Normal',
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS gallery (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT NOT NULL,
                location TEXT NOT NULL,
                image_bytes BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS image_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT UNIQUE NOT NULL,
                location TEXT,
                created_at TEXT,
                embedding TEXT NOT NULL -- JSON-encoded float list
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_query TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                label TEXT NOT NULL, -- 'yes' or 'no'
                response_type TEXT NOT NULL, -- 'fast_answer', 'fast_image', 'ollama', 'error'
                created_at TEXT NOT NULL,
                username TEXT
            )
            """
        )
        conn.commit()


def hash_password(password: str) -> str:
    import hashlib

    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# --- Corporate UI helpers ---
THEMES = {
        "dark": {
                "bg": "#0b1120",
                "panel": "#111827",
                "primary": "#2563eb",
                "accent": "#10b981",
                "text": "#e5e7eb",
                "muted": "#9ca3af",
                "grid": "#1f2937"
        },
        "light": {
                "bg": "#ffffff",
                "panel": "#f8fafc",
                "primary": "#1d4ed8",
                "accent": "#059669",
                "text": "#111827",
                "muted": "#6b7280",
                "grid": "#e5e7eb"
        }
}

def get_theme_colors():
        return THEMES.get(st.session_state.get("theme", "dark"), THEMES["dark"])

def inject_global_styles():
        c = get_theme_colors()
        styles = f"""
        <style>
            html, body, [class^="css"]  {{
                background-color: {c['bg']} !important;
                color: {c['text']} !important;
                font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
            }}
            .stApp {{
                background-color: {c['bg']} !important;
            }}
            .corp-panel {{
                background: {c['panel']};
                border-radius: 12px;
                padding: 16px 20px;
                border: 1px solid {c['grid']};
            }}
            .corp-hero h1 {{
                font-weight: 700;
                letter-spacing: -0.02em;
            }}
            .corp-kpi {{
                background: {c['panel']};
                border-radius: 12px;
                padding: 12px 16px;
                border: 1px solid {c['grid']};
            }}
            .corp-kpi .label {{ color: {c['muted']}; font-size: 12px; }}
            .corp-kpi .value {{ font-size: 22px; font-weight: 600; }}
            .corp-footer {{ color: {c['muted']}; font-size: 12px; margin-top: 16px; }}
            .stButton>button {{
                background: {c['primary']};
                border-radius: 8px; border: none; color: white; font-weight: 600;
            }}
            .stRadio>div {{ gap: 6px; }}
            .stExpander {{ border: 1px solid {c['grid']}; border-radius: 12px; }}

</style>
        """
        st.markdown(styles, unsafe_allow_html=True)


def render_brand_header():
    st.markdown(
        f"<div class='corp-hero'><h1>Redbanded Stink Bug Analytics</h1></div>",
        unsafe_allow_html=True,
    )

def render_top_nav():
    """Top navigation placeholder (intentionally blank)."""
    return


def render_footer():
        st.markdown(
        f"<div class='corp-footer'>© LSU AgCenter & IGLab</div>",
        unsafe_allow_html=True)
def apply_plotly_theme(fig):
    c = get_theme_colors()
    fig.update_layout(
        paper_bgcolor=c["bg"],
        plot_bgcolor=c["panel"],
        font=dict(color=c["text"]),
        xaxis=dict(gridcolor=c["grid"], zerolinecolor=c["grid"]),
        yaxis=dict(gridcolor=c["grid"], zerolinecolor=c["grid"]),
    )

def configure_plotly_template():
    """Create and apply a theme-aware Plotly template globally."""
    c = get_theme_colors()
    template_name = "corp_dark" if st.session_state.get("theme") == "dark" else "corp_light"
    tpl = {
        "layout": {
            "paper_bgcolor": c["bg"],
            "plot_bgcolor": c["panel"],
            "font": {"color": c["text"]},
            "xaxis": {"gridcolor": c["grid"], "zerolinecolor": c["grid"]},
            "yaxis": {"gridcolor": c["grid"], "zerolinecolor": c["grid"]},
            "colorway": [c["primary"], c["accent"], "#f59e0b", "#8b5cf6"],
        }
    }
    pio.templates[template_name] = tpl
    pio.templates.default = template_name


def create_user(username: str, email: str, password: str) -> tuple[bool, str]:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, hash_password(password)),
            )
            conn.commit()
        return True, "Account created successfully. Please log in."
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please pick a different one."


def authenticate_user(username: str, password: str):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, email, password_hash FROM users WHERE username = ?",
            (username,),
        )
        row = cursor.fetchone()
    if not row:
        return None
    user_id, name, email, stored_hash = row
    if hash_password(password) == stored_hash:
        return {"id": user_id, "username": name, "email": email}
    return None


def insert_reports(user, location: str, detection_rows: list[dict]) -> None:
    report_date = datetime.utcnow().strftime("%m::%d::%y")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO reports (user_id, username, report_date, location, image_name, insect_count, priority)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    user["id"],
                    user["username"],
                    report_date,
                    location,
                    row["image_name"],
                    row["count"],
                    row.get("priority", "Normal"),
                )
                for row in detection_rows
            ],
        )
        conn.commit()


def load_reports() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT username, report_date, location, image_name, insect_count, priority FROM reports ORDER BY id DESC",
            conn,
        )
    return df


def location_stats(location: str) -> tuple[int, int]:
    """Gets stats for a location (case-insensitive, normalized by first part before comma)."""
    import re
    
    def normalize_loc(loc: str) -> str:
        loc = loc or ""
        loc = loc.split(",")[0]
        return re.sub(r"[^a-z0-9]", "", loc.casefold())
    
    # Get all locations and find matches
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT location FROM reports WHERE location IS NOT NULL")
        all_locations = [row[0] for row in cursor.fetchall()]
    
    # Find all location variants that match (case-insensitive)
    location_normalized = normalize_loc(location)
    matching_locations = [loc for loc in all_locations if normalize_loc(loc) == location_normalized]
    
    if not matching_locations:
        matching_locations = [location]  # Fallback to exact match
    
    # Sum stats across all matching variants, correcting 0-count images
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        placeholders = ','.join(['?'] * len(matching_locations))
        # Get all reports for these locations
        cursor.execute(
            f"SELECT insect_count, image_name FROM reports WHERE location IN ({placeholders})",
            matching_locations
        )
        rows = cursor.fetchall()
        
        total_images = len(rows)
        total_insects = 0
        for insect_count, image_name in rows:
            # If 0 insects but image exists, count as 1 (detection likely worked but count wasn't saved)
            if insect_count == 0 and image_name:
                total_insects += 1
            else:
                total_insects += insect_count
    
    return total_images or 0, total_insects or 0


def get_global_stats() -> dict:
    """Aggregates global KPIs for dashboard cards."""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*), COALESCE(SUM(insect_count),0) FROM reports")
        total_reports, total_insects = c.fetchone() or (0, 0)
        c.execute("SELECT COUNT(DISTINCT location) FROM reports")
        unique_locations = c.fetchone()[0] or 0
        c.execute("SELECT COUNT(*) FROM gallery")
        gallery_images = c.fetchone()[0] or 0
    alerts = len(st.session_state.get("notifications", []))
    return {
        "reports": total_reports,
        "insects": total_insects,
        "locations": unique_locations,
        "gallery": gallery_images,
        "alerts": alerts,
    }

def _daily_counts(last_days: int = 14) -> dict:
    """Build daily series for reports and insects over last N days."""
    df = load_reports()
    if df.empty:
        today = datetime.utcnow().date()
        days = [today - timedelta(days=i) for i in range(last_days)][::-1]
        zero = [0]*len(days)
        return {"days": days, "images": zero, "insects": zero, "alerts": zero}
    # Parse dates
    df["_dt"] = df["report_date"].apply(_parse_report_date)
    df = df.dropna(subset=["_dt"]).copy()
    df["_day"] = df["_dt"].dt.date
    today = datetime.utcnow().date()
    window = [today - timedelta(days=i) for i in range(last_days)][::-1]
    by_day = df.groupby("_day").agg(images=("image_name","count"), insects=("insect_count","sum")).to_dict("index")
    alerts = len(st.session_state.get("notifications", []))
    return {
        "days": window,
        "images": [by_day.get(d, {}).get("images", 0) for d in window],
        "insects": [by_day.get(d, {}).get("insects", 0) for d in window],
        "alerts": [0]*(len(window)-1)+[alerts],
    }

def render_kpi_sparklines():
    import pandas as pd
    series = _daily_counts(14)
    df = pd.DataFrame({
        "day": series["days"],
        "images": series["images"],
        "insects": series["insects"],
        "alerts": series["alerts"],
    })
    c1, c2, c3 = st.columns(3)
    import plotly.express as px
    def spark(fig):
        fig.update_layout(height=90, margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        apply_plotly_theme(fig)
        return fig
    with c1:
        fig = px.area(df, x="day", y="images")
        st.plotly_chart(spark(fig), use_container_width=True)
    with c2:
        fig = px.area(df, x="day", y="insects")
        st.plotly_chart(spark(fig), use_container_width=True)
    with c3:
        fig = px.area(df, x="day", y="alerts")
        st.plotly_chart(spark(fig), use_container_width=True)


def set_location_priority(location: str, priority: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE reports SET priority = ? WHERE location = ?",
            (priority, location),
        )
        conn.commit()


def send_email_alert(to_email: str, location: str, ratio: float, total_images: int, total_insects: int) -> bool:
    smtp_host = os.getenv("SMTP_HOST")
    print("smtp host", smtp_host)
    smtp_port = os.getenv("SMTP_PORT")
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not all([smtp_host, smtp_port, smtp_user, smtp_password]):
        st.info(
            "Email alert skipped. Please set SMTP_HOST, SMTP_PORT, SMTP_USER, and SMTP_PASSWORD environment variables."
        )
        return False

    msg = EmailMessage()
    msg["Subject"] = f"Stink Bug Alert for {location}"
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg.set_content(
        f"""Hello,

Insect monitoring for {location} exceeded the safety threshold.

- Images analyzed: {total_images}
- Insects counted: {total_insects}
- Density: {int(ratio)} insects per 100 images (approximate calculation)

Please initiate precautionary applications immediately.

Thanks,
LSU Agcenter and IGLab
"""
    )
    try:
        with smtplib.SMTP_SSL(smtp_host, int(smtp_port)) as server:
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        return True
    except Exception as exc:
        st.warning(f"Email alert could not be sent: {exc}")
        return False


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        st.error("Model file yolov8m_cbam_asff_finetuned.pt is missing.")
        st.stop()
    return YOLO(str(MODEL_PATH))

# GroundingDINO helpers (lazy import)
@st.cache_resource
def load_grounding_dino():
    """Load GroundingDINO model with weights."""
    try:
        from groundingdino.util.inference import load_model
        import torch
        
        weights_path = BASE_DIR / "checkpoints/groundingdino_swint_ogc.pth"
        
        if not weights_path.exists():
            st.session_state["groundingdino_error"] = f"Weights not found at {weights_path}"
            return None
        
        # Try to find config file (from cloned repo or package)
        config_paths = [
            BASE_DIR / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            Path(__file__).parent / "groundingdino/config/GroundingDINO_SwinT_OGC.py",
        ]
        
        config_path = None
        for path in config_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path is None:
            # Try package installation path
            try:
                import groundingdino
                pkg_path = Path(groundingdino.__file__).parent
                fallback = pkg_path / "config/GroundingDINO_SwinT_OGC.py"
                if fallback.exists():
                    config_path = fallback
            except:
                pass
        
        if config_path is None:
            st.session_state["groundingdino_error"] = "Config file not found. Clone GroundingDINO repo or install from source."
            return None
            
        # Detect device: CUDA (NVIDIA) or CPU
        # Note: MPS has compatibility issues with GroundingDINO, use CPU on Mac
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
        st.session_state["groundingdino_device"] = device  # Set before loading model
        
        model = load_model(str(config_path), str(weights_path), device=device)
        
        if model is None:
            st.session_state["groundingdino_error"] = "Model failed to load"
            return None
            
        return model
    except Exception as e:
        st.session_state["groundingdino_error"] = str(e)
        return None

def get_grounding_status():
    # Check if already loaded
    if "groundingdino_model" not in st.session_state:
        model = load_grounding_dino()
        st.session_state["groundingdino_model"] = model
    else:
        model = st.session_state["groundingdino_model"]
    
    available = model is not None
    if available and "groundingdino_device" not in st.session_state:
        # Fallback: assume CPU if not recorded
        st.session_state["groundingdino_device"] = "cpu"
    return {
        "available": available, 
        "error": st.session_state.get("groundingdino_error") if not available else None,
        "device": st.session_state.get("groundingdino_device", "unknown") if available else None
    }

def detect_regions_grounding_dino(image_bytes: bytes, prompt: str) -> list:
    """
    Detect regions using GroundingDINO with text prompt.
    Returns list of boxes in [x1, y1, x2, y2] format.
    """
    import sys
    
    # Force flush to see output immediately
    sys.stderr.write("\n" + "="*60 + "\n")
    sys.stderr.write("GroundingDINO Detection Started\n")
    sys.stderr.write("="*60 + "\n")
    sys.stderr.flush()
    
    try:
        from groundingdino.util.inference import predict
        import torch
        from PIL import Image
        import io
        
        model = st.session_state.get("groundingdino_model")
        if model is None:
            model = load_grounding_dino()
            if model is None:
                sys.stderr.write("ERROR: GroundingDINO model not loaded\n")
                sys.stderr.flush()
                st.error("GroundingDINO model failed to load")
                return []
        
        sys.stderr.write("Model loaded successfully\n")
        sys.stderr.flush()
        
        # Load image
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        h, w = image_pil.size[1], image_pil.size[0]
        
        sys.stderr.write(f"Image size: {w}x{h} pixels\n")
        sys.stderr.write(f"Prompt: '{prompt}'\n")
        device = st.session_state.get('groundingdino_device', 'cpu')
        sys.stderr.write(f"Device: {device}\n")
        sys.stderr.write(f"Running prediction with box_threshold=0.10, text_threshold=0.10...\n")
        sys.stderr.flush()
        
        # Use GroundingDINO's preprocessing but run on correct device
        from groundingdino.util.inference import preprocess_caption
        import numpy as np
        
        # Preprocess with error handling
        try:
            import groundingdino.datasets.transforms as T
            transform_func = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_transformed, _ = transform_func(image_pil, None)
            sys.stderr.write("Using GroundingDINO transforms\n")
            sys.stderr.flush()
        except Exception as e:
            # Fallback to simple transform
            sys.stderr.write(f"GroundingDINO transforms failed ({e}), using torchvision fallback\n")
            sys.stderr.flush()
            import torchvision.transforms as T_torch
            simple_transform = T_torch.Compose([
                T_torch.Resize((800, 800)),
                T_torch.ToTensor(),
                T_torch.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_transformed = simple_transform(image_pil)
        
        # Preprocess caption
        caption = preprocess_caption(caption=prompt)
        
        # Run model on correct device
        model.eval()
        image_transformed = image_transformed.to(device)
        
        with torch.no_grad():
            outputs = model(image_transformed[None], captions=[caption])
        
        # Process outputs
        prediction_logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        prediction_boxes = outputs["pred_boxes"][0]  # (nq, 4)
        
        # Filter by thresholds
        box_threshold = 0.10
        text_threshold = 0.10
        
        logits_filt = prediction_logits.cpu().clone()
        boxes_filt = prediction_boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        
        # Get phrases
        from groundingdino.util.utils import get_phrases_from_posmap
        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)
        phrases = []
        for logit in logits_filt:
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            phrases.append(pred_phrase)
        
        boxes = boxes_filt
        logits = logits_filt.max(dim=1)[0]
        
        sys.stderr.write(f"Raw detections: {len(boxes)}\n")
        sys.stderr.flush()
        
        if len(boxes) > 0:
            sys.stderr.write(f"Confidences: {[f'{l:.3f}' for l in logits.tolist()]}\n")
            sys.stderr.write(f"Phrases: {phrases}\n")
            sys.stderr.flush()
        else:
            sys.stderr.write("  NO DETECTIONS - Trying alternative prompts...\n")
            sys.stderr.flush()
            # Try alternative prompts with same manual inference
            for alt_prompt in ["insect", "bug", "green insect", "shield bug"]:
                sys.stderr.write(f"  Trying: '{alt_prompt}'\n")
                sys.stderr.flush()
                
                # Manual inference for alternative prompt
                alt_caption = preprocess_caption(caption=alt_prompt)
                with torch.no_grad():
                    alt_outputs = model(image_transformed[None], captions=[alt_caption])
                
                alt_logits_raw = alt_outputs["pred_logits"].sigmoid()[0]
                alt_boxes_raw = alt_outputs["pred_boxes"][0]
                
                alt_logits_filt = alt_logits_raw.cpu().clone()
                alt_boxes_filt = alt_boxes_raw.cpu().clone()
                alt_mask = alt_logits_filt.max(dim=1)[0] > box_threshold
                alt_logits_filt = alt_logits_filt[alt_mask]
                alt_boxes_filt = alt_boxes_filt[alt_mask]
                
                alt_tokenized = tokenizer(alt_caption)
                alt_phrases = []
                for logit in alt_logits_filt:
                    pred_phrase = get_phrases_from_posmap(logit > text_threshold, alt_tokenized, tokenizer)
                    alt_phrases.append(pred_phrase)
                
                if len(alt_boxes_filt) > 0:
                    sys.stderr.write(f"  Found {len(alt_boxes_filt)} with '{alt_prompt}'!\n")
                    sys.stderr.flush()
                    boxes = alt_boxes_filt
                    logits = alt_logits_filt.max(dim=1)[0]
                    phrases = alt_phrases
                    break
        
        # Convert boxes from normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]
        boxes_list = []
        for i, box in enumerate(boxes):
            cx, cy, box_w, box_h = box.tolist()
            x1 = max(0, (cx - box_w / 2) * w)
            y1 = max(0, (cy - box_h / 2) * h)
            x2 = min(w, (cx + box_w / 2) * w)
            y2 = min(h, (cy + box_h / 2) * h)
            
            box_pixel_w = x2 - x1
            box_pixel_h = y2 - y1
            
            # Filter out tiny boxes (< 10 pixels)
            if box_pixel_w > 10 and box_pixel_h > 10:
                boxes_list.append([x1, y1, x2, y2])
                sys.stderr.write(f"Box {i}: [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] {box_pixel_w:.0f}x{box_pixel_h:.0f}px conf={logits[i]:.3f}\n")
            else:
                sys.stderr.write(f"Box {i}: {box_pixel_w:.0f}x{box_pixel_h:.0f}px REJECTED (too small)\n")
        sys.stderr.flush()
        
        sys.stderr.write(f"Final: {len(boxes_list)} boxes kept\n")
        sys.stderr.write("="*60 + "\n\n")
        sys.stderr.flush()
        
        # Also show in UI for debugging
        if len(boxes) == 0:
            st.warning(f"🔍 GroundingDINO: Image {w}x{h}px, found 0 detections for '{prompt}'. Check terminal for details.")
        
        return boxes_list
    except Exception as e:
        sys.stderr.write(f"GroundingDINO ERROR: {e}\n")
        sys.stderr.flush()
        print(f"GroundingDINO ERROR: {e}")
        st.error(f"GroundingDINO error: {e}")
        import traceback
        traceback.print_exc()
        cx, cy, box_w, box_h = box.tolist()
        x1 = max(0, (cx - box_w / 2) * w)
        y1 = max(0, (cy - box_h / 2) * h)
        x2 = min(w, (cx + box_w / 2) * w)
        y2 = min(h, (cy + box_h / 2) * h)
        
        box_pixel_w = x2 - x1
        box_pixel_h = y2 - y1
        
        # Filter out tiny boxes (< 10 pixels)
        if box_pixel_w > 10 and box_pixel_h > 10:
            boxes_list.append([x1, y1, x2, y2])
            print(f"  - Box {i}: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}] size={box_pixel_w:.0f}x{box_pixel_h:.0f}px conf={logits[i]:.3f}")
        else:
            print(f"  - Box {i}: REJECTED (too small: {box_pixel_w:.0f}x{box_pixel_h:.0f}px)")
    
        print(f"GroundingDINO final: {len(boxes_list)} boxes kept (after filtering tiny boxes)")
        return boxes_list
    except Exception as e:
        print(f"GroundingDINO error: {e}")
        import traceback
        traceback.print_exc()
        import traceback
        traceback.print_exc()
        return []

def compute_iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    areaA = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    areaB = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
    denom = areaA + areaB - interArea
    return (interArea / denom) if denom > 0 else 0.0

def filter_yolo_boxes_by_text(yolo_boxes: list, text_boxes: list, iou_threshold: float) -> list:
    if not text_boxes:
        return yolo_boxes
    filtered = []
    for b in yolo_boxes:
        if any(compute_iou(b, tb) >= iou_threshold for tb in text_boxes):
            filtered.append(b)
    return filtered



def count_insects(image_bytes: bytes, use_sam2_verification: bool = False, text_filter_enabled: bool = False, text_prompt: str = "", iou_threshold: float = 0.5, yolo_conf_threshold: float = 0.5) -> tuple[int, bytes, dict]:
    """
    Count insects with optional SAM 2 verification and optional text-prompt filtering via GroundingDINO.
    
    Args:
        image_bytes: Raw image bytes
        use_sam2_verification: If True, verify YOLO boxes with SAM 2
        text_filter_enabled: If True, filter YOLO boxes by text-prompt regions (GroundingDINO)
        text_prompt: Text prompt for GroundingDINO filtering
        iou_threshold: IoU threshold for text-prompt box matching
        yolo_conf_threshold: Confidence threshold to filter out low-confidence YOLO boxes BEFORE SAM2
    
    Returns:
        (count, annotated_bytes, metadata)
    """
    model = load_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model(image, verbose=False)
    if not results:
        return 0, image_bytes, {"verified": False}

    result = results[0]
    boxes = result.boxes
    yolo_original_count = int(boxes.shape[0]) if boxes is not None else 0
    
    # Extract confidences and boxes
    confidences = boxes.conf.cpu().numpy().tolist() if boxes is not None else []
    yolo_boxes = boxes.xyxy.cpu().numpy().tolist() if boxes is not None else []
    
    # Pre-SAM2 validation: filter by confidence threshold
    pre_sam2_boxes = []
    pre_sam2_confidences = []
    for i, (box, conf) in enumerate(zip(yolo_boxes, confidences)):
        if conf >= yolo_conf_threshold:
            pre_sam2_boxes.append(box)
            pre_sam2_confidences.append(conf)
    
    yolo_count_after_conf = len(pre_sam2_boxes)
    yolo_boxes = pre_sam2_boxes

    # Initialize metadata early
    metadata = {
        "yolo_original_count": yolo_original_count,
        "yolo_count_after_confidence_filter": yolo_count_after_conf,
        "yolo_count": yolo_count_after_conf,
        "verified": False,
        "text_filter_applied": bool(text_filter_enabled and text_prompt.strip()),
        "text_prompt": text_prompt.strip() if text_prompt else "",
        "iou_threshold": float(iou_threshold),
        "text_boxes_count": 0,
        "grounding_dino_fallback": False,
        "yolo_conf_threshold": float(yolo_conf_threshold),
        "boxes_filtered_by_confidence": yolo_original_count - yolo_count_after_conf
    }

    # Optional text-prompt filtering
    text_boxes = []
    if text_filter_enabled and text_prompt.strip():
        status = get_grounding_status()
        if status["available"]:
            try:
                text_boxes = detect_regions_grounding_dino(image_bytes, text_prompt.strip())
                metadata["text_boxes_count"] = len(text_boxes)
                
                # Apply filtering based on GroundingDINO results
                if text_boxes:
                    # Found matching regions - keep only YOLO boxes that overlap
                    filtered = filter_yolo_boxes_by_text(yolo_boxes, text_boxes, iou_threshold)
                    metadata["boxes_filtered_by_text"] = len(yolo_boxes) - len(filtered)
                    yolo_boxes = filtered
                    metadata["grounding_dino_fallback"] = False
                else:
                    metadata["grounding_dino_fallback"] = True
                    print(f"Warning: GroundingDINO found 0 '{text_prompt}' regions. Keeping {yolo_count_after_conf} YOLO boxes (fallback mode - may include false positives).")
            except Exception as e:
                # Error in GroundingDINO - skip text filtering, keep confidence-passed boxes
                metadata["grounding_dino_error"] = str(e)
                print(f"GroundingDINO error: {e}. Skipping text filtering, keeping {yolo_count_after_conf} confidence-passed boxes.")
        else:
            # GroundingDINO not available
            pass
    # If text_filter_enabled is False, skip all GroundingDINO processing entirely

    yolo_count = len(yolo_boxes)
    metadata["yolo_count"] = yolo_count

    # SAM 2 Verification (no judge mode): only validate YOLO boxes when enabled
    if use_sam2_verification and yolo_count > 0:
        try:
            verifier = get_sam2_verifier()
            verification_results = verifier.verify_yolo_boxes(image_bytes, yolo_boxes)
            metadata["verified"] = True
            metadata["yolo_verified_count"] = verification_results['verified_count']
            metadata["annotation_quality"] = verification_results['annotation_quality']
            metadata["quality_issues"] = verification_results['quality_issues']
            metadata["avg_confidence"] = verification_results.get('avg_confidence', 0.0)
            count = verification_results['verified_count']
            metadata["verification_note"] = f"SAM 2 verified {count}/{yolo_count} detections"
        except Exception as e:
            count = yolo_count
            metadata["verification_error"] = str(e)
    else:
        count = yolo_count

    # Annotate image: use filtered boxes if text filter applied; otherwise use model's plot
    if metadata["text_filter_applied"] and yolo_boxes:
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        for b in yolo_boxes:
            x1, y1, x2, y2 = map(float, b)
            draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)
        buffer = io.BytesIO()
        annotated_image.save(buffer, format="PNG")
        annotated_bytes = buffer.getvalue()
    else:
        annotated_array = result.plot()  # BGR numpy array
        annotated_image = Image.fromarray(annotated_array[..., ::-1])
        buffer = io.BytesIO()
        annotated_image.save(buffer, format="PNG")
        annotated_bytes = buffer.getvalue()

    return count, annotated_bytes, metadata


def ensure_session_defaults():
    defaults = {
        "auth_user": None,
        "menu_choice": "Homepage",
        "theme": "dark",
        "latest_detections": [],
        "notifications": [],
        "processed_upload_signature": None,  # legacy, kept for compatibility
        "detection_cache": {},
        "chat_popup_open": False,  # For floating chat button toggle
        "messages": [],  # Chat history
        "ollama_client": None,  # Ollama client for chat
        "agent_mode": False,  # Enable advanced agentic mode
        "show_agent_reasoning": False,  # Show chain-of-thought
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    
    # Initialize Ollama client if not already done
    if st.session_state["ollama_client"] is None:
        try:
            from ollama import Client
            st.session_state["ollama_client"] = Client(host='http://localhost:11434')
        except Exception as e:
            st.warning(f"Could not connect to Ollama: {e}")
            st.session_state["ollama_client"] = None


def logout():
    st.session_state["auth_user"] = None
    st.session_state["menu_choice"] = "Homepage"
    st.session_state["latest_detections"] = []
    st.rerun()


def threshold_check(location: str, user_email: str) -> None:
    total_images, total_insects = location_stats(location)
    if total_images == 0:
        return
    ratio = (total_insects / total_images) * 100

    # SAM 2 VERIFICATION: Use when approaching threshold (12-16 range)
    approaching_threshold = 12 <= ratio < THRESHOLD_PER_100
    use_sam2 = approaching_threshold or ratio >= THRESHOLD_PER_100

    verification_note = ""
    if use_sam2:
        verification_note = " [SAM 2 verified]"
        st.info(f"Approaching threshold - activating SAM 2 verification for {location}")

    # User rule:
    # 1) When images are 100 and count is at least 16  -> 16 insects per 100 images
    # 2) When count is already 16 even if images < 100 -> early warning
    meets_threshold = (total_images >= 100 and ratio >= THRESHOLD_PER_100) or (
        total_images < 100 and total_insects >= THRESHOLD_PER_100
    )

    # Always log a notification so the user sees that the location was updated
    base_message = (
        f"Location '{location}' updated: {total_insects} insects over {total_images} images "
        f"({int(ratio)} insects / 100 images){verification_note}."
    )

    if meets_threshold:
        set_location_priority(location, "Priority")
        message = "Priority alert: " + base_message
        st.toast(message)
        send_email_alert(user_email, location, ratio, total_images, total_insects)
    else:
        set_location_priority(location, "Normal")
        message = "Below threshold: " + base_message

    st.session_state["notifications"].insert(
        0,
        {
            "timestamp": datetime.utcnow().strftime("%H:%M:%S"),
            "message": message,
            "location": location,
            "ratio": ratio,
            "sam2_verified": use_sam2,
        },
    )


def update_gallery(location: str, detections: list[dict]) -> None:
    timestamp = datetime.utcnow().strftime("%m/%d %H:%M")

    # Update in-memory gallery for the current session
    session_entries = [
        {
            "image_name": det["image_name"],
            "location": location,
            "annotated": det["annotated"],
            "timestamp": timestamp,
        }
        for det in detections
    ]
    existing = st.session_state.get("gallery", [])
    st.session_state["gallery"] = (session_entries + existing)[:MAX_GALLERY_ITEMS]

    # Persist annotated images so they are available after logout/login
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO gallery (image_name, location, image_bytes, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    det["image_name"],
                    location,
                    det["annotated"],
                    datetime.utcnow().isoformat(timespec="seconds"),
                )
                for det in detections
            ],
        )
        conn.commit()


def load_gallery(limit: int = MAX_GALLERY_ITEMS) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Get the FIRST/original entry for each unique image_name (MIN(id) = first upload)
        if limit is None:
            cursor.execute(
                """
                SELECT image_name, location, image_bytes, created_at
                FROM gallery
                WHERE id IN (
                    SELECT MIN(id) FROM gallery GROUP BY image_name
                )
                ORDER BY id DESC
                """
            )
        else:
            cursor.execute(
                """
                SELECT image_name, location, image_bytes, created_at
                FROM gallery
                WHERE id IN (
                    SELECT MIN(id) FROM gallery GROUP BY image_name
                )
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
        rows = cursor.fetchall()

    entries: list[dict] = []
    for image_name, location, image_bytes, created_at in rows:
        entries.append(
            {
                "image_name": image_name,
                "location": location,
                "annotated": image_bytes,
                "timestamp": datetime.fromisoformat(created_at).strftime("%m/%d %H:%M"),
            }
        )
    return entries


# --- Local Retrieval Index (image embeddings + search) ---
def compute_image_embedding(image_bytes: bytes) -> list[float]:
    """Computes a simple HSV color histogram embedding for an image.
    Returns a normalized vector (length 96: 32 bins per H, S, V).
    """
    import cv2
    import numpy as np
    # Decode image bytes to BGR for OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        # Fallback via PIL
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bins = 32
    h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    vec = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()]).astype(np.float32)
    # Normalize to unit length
    norm = np.linalg.norm(vec) or 1.0
    vec = (vec / norm).tolist()
    return vec


def refresh_image_index() -> int:
    """Creates/updates embeddings for images present in the gallery but missing in image_index.
    Returns number of newly indexed images.
    """
    import json
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        # Load existing indexed image names
        c.execute("SELECT image_name FROM image_index")
        existing = {row[0] for row in c.fetchall()}

        # Fetch all gallery rows
        c.execute("SELECT image_name, location, image_bytes, created_at FROM gallery ORDER BY id DESC")
        rows = c.fetchall()
        new_count = 0
        for image_name, location, image_bytes, created_at in rows:
            if image_name in existing:
                continue
            try:
                emb = compute_image_embedding(image_bytes)
                c.execute(
                    "INSERT OR REPLACE INTO image_index (image_name, location, created_at, embedding) VALUES (?, ?, ?, ?)",
                    (image_name, location, created_at, json.dumps(emb)),
                )
                new_count += 1
            except Exception:
                # Skip problematic image
                continue
        conn.commit()
        return new_count


def load_image_index() -> list[dict]:
    import json
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT image_name, location, created_at, embedding FROM image_index")
        out = []
        for image_name, location, created_at, emb_json in c.fetchall():
            try:
                emb = json.loads(emb_json)
            except Exception:
                emb = []
            out.append({
                "image_name": image_name,
                "location": location,
                "created_at": created_at,
                "embedding": emb,
            })
        return out


def is_valid_word_match(text: str, query: str) -> bool:
    """Validate that query matches only real words in text, not random character sequences.
    Uses word boundary regex to ensure proper word matching.
    e.g., 'LSU' matches 'LSU, Alabama' but NOT 'KLSUQCK' in filenames
    """
    import re
    if not text or not query:
        return False
    
    # Create word boundary pattern - matches query as whole word(s)
    # \b ensures word boundaries
    pattern = r'\b' + re.escape(query.lower()) + r'\b'
    match = re.search(pattern, text.lower())
    return match is not None


def search_images_by_text(query: str, top_k: int = 12) -> list[dict]:
    """Case-insensitive word-boundary search across image_name and location.
    Returns gallery entries with scores (deduplicated by image_name, using FIRST location).
    Only matches complete words, not random character sequences in filenames.
    """
    q = (query or "").lower().strip()
    if not q:
        return []
    # Build gallery lookup - get EARLIEST entry for each unique image_name (original location)
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        # Query: Get the FIRST entry for each image_name (MIN(id) = original upload)
        c.execute("""
            SELECT image_name, location, image_bytes, created_at 
            FROM gallery 
            WHERE id IN (
                SELECT MIN(id) FROM gallery GROUP BY image_name
            )
            ORDER BY id DESC
        """)
        rows = c.fetchall()
    results = []
    for image_name, location, image_bytes, created_at in rows:
        name_l = (image_name or "").lower()
        loc_l = (location or "").lower()
        score = 0
        
        # Check for WORD BOUNDARY matches only (not substring matches in random filenames)
        if is_valid_word_match(name_l, q):
            score += 2
        if is_valid_word_match(loc_l, q):
            score += 3
        
        # Also check token overlap (whole words)
        tokens = set(q.split())
        for t in tokens:
            if is_valid_word_match(name_l, t):
                score += 1
            if is_valid_word_match(loc_l, t):
                score += 1
        
        if score > 0:
            results.append({
                "image_name": image_name,
                "location": location,
                "image_bytes": image_bytes,
                "created_at": created_at,
                "score": score,
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def search_images_by_image(image_bytes: bytes, top_k: int = 12) -> list[dict]:
    """Find similar images using cosine similarity of HSV histogram embeddings."""
    import numpy as np
    query_emb = np.array(compute_image_embedding(image_bytes), dtype=np.float32)
    index = load_image_index()
    if not index:
        return []
    # Build gallery bytes map for display
    gallery_items = {item["image_name"]: item for item in load_gallery(limit=10000)}
    def cosine(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / denom)
    scored = []
    for rec in index:
        sim = cosine(query_emb, rec["embedding"]) if rec["embedding"] else 0.0
        g = next((x for x in gallery_items if x == rec["image_name"]), None)
        item = gallery_items.get(rec["image_name"]) if g is None else gallery_items[g]
        if item:
            scored.append({
                "image_name": rec["image_name"],
                "location": rec["location"],
                "image_bytes": item["annotated"],
                "created_at": rec["created_at"],
                "score": sim,
            })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def render_search():
    """Renders Text/Image search with local similarity index."""
    st.subheader("Search Images")
    # Ensure index is up-to-date
    new_indexed = refresh_image_index()
    if new_indexed:
        st.caption(f"Indexed {new_indexed} new images.")

    tab_text, tab_image = st.tabs(["Text Search", "Image Similarity"])

    with tab_text:
        q = st.text_input("Search by text (location, name)", key="search_text_query")
        top_k = st.slider("Results", min_value=3, max_value=24, value=12, step=3, key="search_text_topk")
        if q:
            results = search_images_by_text(q, top_k=top_k)
            if not results:
                st.info("No matches.")
            else:
                cols = st.columns(3)
                for i, res in enumerate(results):
                    with cols[i % 3]:
                        st.image(res["image_bytes"], caption=f"{res['image_name']} — {res['location']} (score {res['score']})", use_container_width=True)

    with tab_image:
        uploaded = st.file_uploader("Upload an image to find similar", type=["jpg", "jpeg", "png"], key="search_image_upload")
        top_k_img = st.slider("Results", min_value=3, max_value=24, value=12, step=3, key="search_image_topk")
        if uploaded:
            bytes_data = uploaded.getvalue()
            results = search_images_by_image(bytes_data, top_k=top_k_img)
            if not results:
                st.info("No similar images found or index empty.")
            else:
                cols = st.columns(3)
                for i, res in enumerate(results):
                    with cols[i % 3]:
                        st.image(res["image_bytes"], caption=f"{res['image_name']} — {res['location']} (sim {res['score']:.2f})", use_container_width=True)


def render_slideshow(title: str, entries: list[dict], slider_key: str):
    if not entries:
        st.caption("No images to display yet.")
        return
    st.subheader(title)
    counter = st_autorefresh(interval=10_000, key=f"{slider_key}_refresh")
    index = counter % len(entries)
    entry = entries[index]
    st.image(
        entry["annotated"],
        caption=f"{entry['image_name']} — {entry.get('location', 'Unknown location')}",
        use_container_width=True,
    )
    st.caption(f"Captured at: {entry.get('timestamp', '—')}")


def render_professional_pagination(current_page: int, total_pages: int, session_key: str):
    """Render a professional pagination control with numbered buttons."""
    if total_pages <= 1:
        return
    
    # Calculate which page buttons to show
    max_buttons = 7
    if total_pages <= max_buttons:
        page_range = list(range(total_pages))
    else:
        if current_page < 3:
            page_range = list(range(min(5, total_pages))) + [None] + [total_pages - 1]
        elif current_page > total_pages - 4:
            page_range = [0, None] + list(range(max(5, total_pages - 5), total_pages))
        else:
            page_range = [0, None] + list(range(current_page - 1, min(current_page + 2, total_pages))) + [None, total_pages - 1]
    
    c = get_theme_colors()
    pagination_html = f"""
    <div style='display:flex; justify-content:center; gap:8px; align-items:center; margin:20px 0; flex-wrap:wrap;'>
    """
    
    # Previous button
    if current_page > 0:
        pagination_html += f"""
        <button style='padding:8px 12px; border:1px solid {c["grid"]}; background:{c["primary"]}; color:white; border-radius:6px; cursor:pointer; font-weight:500;' onclick="document.querySelector('[data-page={current_page-1}]').click();">← Previous</button>
        """
    
    # Page buttons
    for page in page_range:
        if page is None:
            pagination_html += f"<span style='color:{c['muted']};'>...</span>"
        else:
            is_current = page == current_page
            bg = c["primary"] if is_current else c["panel"]
            border = f"2px solid {c['primary']}" if is_current else f"1px solid {c['grid']}"
            pagination_html += f"""
            <button data-page="{page}" style='padding:8px 12px; border:{border}; background:{bg}; color:white; border-radius:6px; cursor:pointer; font-weight:{"600" if is_current else "500"}; min-width:40px;' onclick="document.querySelector('[data-page-select={page}]').click();">{page + 1}</button>
            """
    
    # Next button
    if current_page < total_pages - 1:
        pagination_html += f"""
        <button style='padding:8px 12px; border:1px solid {c["grid"]}; background:{c["primary"]}; color:white; border-radius:6px; cursor:pointer; font-weight:500;' onclick="document.querySelector('[data-page={current_page+1}]').click();">Next →</button>
        """
    
    pagination_html += f"<span style='color:{c['muted']}; margin-left:12px;'>Page {current_page + 1} of {total_pages}</span>"
    pagination_html += "</div>"
    
    st.markdown(pagination_html, unsafe_allow_html=True)


def render_location_bar_chart():
    """Renders an interactive bar chart showing stinkbug counts by location."""
    reports = get_reports_data()
    
    if not reports:
        st.info("No data available yet. Upload images and save reports to see statistics.")
        return
    
    # Build location statistics
    location_data = {}
    for report in reports:
        loc = report.get('location', '').strip()
        if loc:
            if loc not in location_data:
                location_data[loc] = {'images': 0, 'insects': 0}
            location_data[loc]['images'] += 1
            # Count insects: use stored count, but if 0 and image exists, count as at least 1
            insect_count = report.get('insect_count', 0)
            if insect_count == 0 and report.get('image_name'):
                insect_count = 1
            location_data[loc]['insects'] += insect_count
    
    if not location_data:
        st.info("No location data available yet.")
        return
    
    # Group data by normalized location (case-insensitive) for unique bars
    import pandas as pd
    from collections import defaultdict
    import re

    def normalize_loc(loc: str) -> str:
        loc = loc or ""
        loc = loc.split(",")[0]  # collapse variants like "Butwal, Nepal" vs "Butwal"
        return re.sub(r"[^a-z0-9]", "", loc.casefold())

    location_reports = defaultdict(list)
    # Track the most specific (longest) original location string per normalized key
    location_strings = {}
    for loc, data in location_data.items():
        key = normalize_loc(loc)
        location_reports[key].append({
            'insects': data['insects'],
            'images': data['images']
        })
        if key not in location_strings:
            location_strings[key] = loc
        else:
            # Prefer the longer name (e.g., "Butwal, Nepal" over "Butwal")
            if len(loc) > len(location_strings[key]):
                location_strings[key] = loc

    df_list = []
    for key, reports in location_reports.items():
        total_insects = sum(r['insects'] for r in reports)
        total_images = sum(r['images'] for r in reports)
        df_list.append({
            'Location': location_strings.get(key, key) or "Unknown",
            'Stinkbugs Detected': total_insects,
            'Images Analyzed': total_images,
            'Report Count': len(reports)
        })

    df = pd.DataFrame(df_list)
    df = df.sort_values('Stinkbugs Detected', ascending=False)
    
    # Create stacked bar chart
    fig = px.bar(
        df,
        x='Location',
        y='Stinkbugs Detected',
        title='Total Stinkbug Detections by Location (Unique Cities)',
        labels={'Stinkbugs Detected': 'Number of Stinkbugs'},
        color='Report Count',
        color_continuous_scale='Reds',
        text='Stinkbugs Detected',
        hover_data={'Images Analyzed': True, 'Report Count': True}
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title='Location (Unique)',
        yaxis_title='Number of Stinkbugs Detected',
        height=500,
        xaxis={'categoryorder': 'total descending'}
    )
    apply_plotly_theme(fig)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Location-based image gallery
    st.markdown("**Click on a location below to view images from that location:**")
    selected_location = st.selectbox(
        "Select a location to view images",
        options=df['Location'].tolist(),
        label_visibility="collapsed"
    )
    
    if selected_location:
        # Fetch all images from gallery for selected location (case-insensitive)
        # Get all unique locations from gallery
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT location FROM gallery")
            all_locations = [row[0] for row in cursor.fetchall() if row[0]]
        
        # Normalize and find all matching location variants
        import re
        def normalize_loc(loc: str) -> str:
            loc = loc or ""
            loc = loc.split(",")[0]
            return re.sub(r"[^a-z0-9]", "", loc.casefold())
        
        selected_normalized = normalize_loc(selected_location)
        matching_locations = [loc for loc in all_locations if normalize_loc(loc) == selected_normalized]
        
        if not matching_locations:
            matching_locations = [selected_location]  # Fallback to exact match
        
        # Fetch images for all matching location variants
        rows = []
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            placeholders = ','.join(['?'] * len(matching_locations))
            cursor.execute(
                f"SELECT image_name, image_bytes, created_at FROM gallery WHERE location IN ({placeholders}) ORDER BY created_at DESC",
                matching_locations
            )
            rows = cursor.fetchall()
        
        if rows:
            st.markdown(f"**Images from {selected_location}** ({len(rows)} images)")
            
            # Display all images in 3-column grid
            cols = st.columns(3)
            for i, (image_name, image_bytes, created_at) in enumerate(rows):
                with cols[i % 3]:
                    st.image(image_bytes, caption=f"{image_name[:20]}...", use_container_width=True)
                    st.caption(created_at[:10] if created_at else "Unknown date")
        else:
            st.info(f"No images yet for {selected_location}")



def render_home():
    render_top_nav()
    render_brand_header()

    # KPI cards
    stats = get_global_stats()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='corp-kpi'><div class='label'>Reports</div><div class='value'>{stats['reports']}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='corp-kpi'><div class='label'>Total Insects</div><div class='value'>{stats['insects']}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='corp-kpi'><div class='label'>Locations</div><div class='value'>{stats['locations']}</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='corp-kpi'><div class='label'>Gallery Images</div><div class='value'>{stats['gallery']}</div></div>", unsafe_allow_html=True)

    st.divider()

    # Multi-page tabs instead of scrolling
    tab_recent, tab_analysis, tab_alerts = st.tabs(["Recent Detections", "Location Analysis", "Alerts"])
    
    # PAGE 1: Recent Detections
    with tab_recent:
        all_gallery = load_gallery(limit=None)  # Get all images
        if all_gallery:
            cols = st.columns(3)
            for i, img_data in enumerate(all_gallery):
                with cols[i % 3]:
                    st.image(
                        img_data["annotated"],
                        caption=f"{img_data['image_name'][:15]}... ({img_data['location']})",
                        use_container_width=True
                    )
                    st.caption(f"Captured: {img_data['timestamp']}")
        else:
            st.info("No images captured yet.")
    
    # PAGE 2: Location Analysis
    with tab_analysis:
        render_location_bar_chart()
    
    # PAGE 3: Alerts
    with tab_alerts:
        notifications = st.session_state.get("notifications", [])
        if notifications:
            for note in notifications[:10]:
                st.info(f"[{note['timestamp']}] {note['message']}")
        else:
            st.caption("No alerts yet. Alerts will appear here when thresholds are exceeded.")

    render_footer()


def render_feedback_analytics():
    """Dashboard showing feedback statistics and export functionality."""
    render_top_nav()
    st.title("Feedback Analytics")
    
    # Debug: Check if table exists and show raw count
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM feedback")
            count = cursor.fetchone()[0]
            
            # Show session state feedback
            session_feedback = st.session_state.get("feedback", [])
            
            df = pd.read_sql_query("SELECT * FROM feedback ORDER BY created_at DESC", conn)
    except Exception as e:
        st.error(f"Error reading feedback table: {e}")
        st.info("Try restarting the app to create the feedback table.")
        return
    
    if df.empty:
        st.warning("  No feedback in database yet!")
        st.info("Ask questions in Chat and click 👍/👎 buttons. Check terminal for debug messages.")
        
        # Test button to verify save function works
        st.divider()
        st.subheader("🧪 Test Feedback Save")
        if st.button("Test Save Feedback"):
            try:
                save_feedback_to_db(
                    user_query="Test question",
                    assistant_response="Test answer",
                    label="yes",
                    response_type="test"
                )
                st.success("Test feedback saved! Refresh page to see it.")
                st.rerun()
            except Exception as e:
                st.error(f"Test failed: {e}")
        
        # Show instructions
        with st.expander("🐛 Troubleshooting"):
            st.markdown("""
            **If buttons don't work:**
            1. Restart the Streamlit app completely (Ctrl+C, then `streamlit run app.py`)
            2. Go to chat
            3. Ask a question
            4. **Look for the feedback buttons below the answer** (👉 Was this helpful? 👍 👎)
            5. Click the 👍 or 👎 button
            6. Check terminal output
            7. Come back to Feedback Analytics
            
            **Buttons should appear:**
            - Immediately after fast answers (location counts, summaries)
            - Immediately after image responses
            - Immediately after AI (Ollama) responses
            
            **If you still see nothing:**
            - Share the terminal output with error messages
            - Try the "Test Save Feedback" button above
            """)
        return
    
    # KPIs
    total = len(df)
    helpful = len(df[df['label'] == 'yes'])
    not_helpful = len(df[df['label'] == 'no'])
    helpful_rate = (helpful / total * 100) if total > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='corp-kpi'><div class='label'>Total Feedback</div><div class='value'>{total}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='corp-kpi'><div class='label'>Helpful (👍)</div><div class='value'>{helpful}</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='corp-kpi'><div class='label'>Not Helpful (👎)</div><div class='value'>{not_helpful}</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='corp-kpi'><div class='label'>Helpful Rate</div><div class='value'>{helpful_rate:.1f}%</div></div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Feedback by response type
    st.subheader("Feedback by Response Type")
    by_type = df.groupby(['response_type', 'label']).size().unstack(fill_value=0)
    if not by_type.empty:
        fig = px.bar(by_type, barmode='group', title='Helpful vs Not Helpful by Response Type')
        st.plotly_chart(fig, use_container_width=True)
    
    # Feedback over time
    st.subheader("Feedback Trend Over Time")
    df['date'] = pd.to_datetime(df['created_at']).dt.date
    by_date = df.groupby(['date', 'label']).size().unstack(fill_value=0)
    if not by_date.empty:
        fig = px.line(by_date, title='Feedback Over Time')
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Export training data
    st.subheader("Export Training Data")
    st.markdown("Download thumbs-up Q&A pairs for fine-tuning or analysis:")
    
    col1, col2, col3 = st.columns(3)
    
   
    with col2:
        if st.button("📥 Download as CSV"):
            helpful_df = df[df['label'] == 'yes'][['user_query', 'assistant_response', 'response_type', 'created_at', 'username']]
            
            if helpful_df.empty:
                st.warning("No thumbs-up feedback to export yet.")
            else:
                csv_content = helpful_df.to_csv(index=False)
                
                st.download_button(
                    label="💾 Download CSV",
                    data=csv_content,
                    file_name=f"feedback_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                st.success(f"Ready to download {len(helpful_df)} Q&A pairs? Click Download CSV above.")
    
    
    # Recent feedback table
    st.divider()
    st.subheader("Recent Feedback")
    
    # Show most recent 20
    recent = df.head(20)[['user_query', 'assistant_response', 'label', 'response_type', 'created_at']]
    recent['user_query'] = recent['user_query'].str[:50] + '...'
    recent['assistant_response'] = recent['assistant_response'].str[:50] + '...'
    st.dataframe(recent, use_container_width=True)
    
    render_footer()


def render_capture():
    st.subheader("Capture & Analyze RBSB Images")

    # Check SAM 2 availability
    verifier = get_sam2_verifier()
    sam2_status = verifier.get_status()

    # SAM 2 Verification Toggle (shows when approaching threshold)
    with st.expander("Advanced Options", expanded=False):
        # Show SAM 2 status
        if sam2_status['available']:
            st.success(f"SAM 2 loaded on {sam2_status['device']}")
        else:
            error_msg = sam2_status['error'] or "Unknown error"
            st.warning(f"SAM 2 not available: {error_msg}")
            st.info("App will use YOLO-only counting (still accurate!)")

        use_sam2 = st.checkbox(
            "Enable SAM 2 Verification (more accurate, slower)",
            value=st.session_state.get("use_sam2_verification", False),
            help="Activate SAM 2 to double-check YOLO boxes with instance segmentation for higher accuracy.",
            key="sam2_toggle",
            disabled=not sam2_status['available']
        )
        st.session_state["use_sam2_verification"] = use_sam2

        if use_sam2 and sam2_status['available']:
            st.info("🔬 SAM 2 verification active: YOLO detections will be validated with segmentation for higher accuracy.")

        # YOLO Confidence Threshold (pre-SAM2 filter)
        st.divider()
        st.markdown("**YOLO Detection Filter**")
        yolo_conf_threshold = st.slider(
            "YOLO confidence threshold (filters before SAM 2)",
            min_value=0.3, max_value=0.95, value=float(st.session_state.get("yolo_conf_threshold", 0.5)), step=0.05,
            help="Lower = more detections but possibly more false positives (e.g., leaves). Higher = fewer detections but higher quality.",
            key="yolo_conf_slider"
        )
        st.session_state["yolo_conf_threshold"] = yolo_conf_threshold
        st.caption(f"Boxes below {yolo_conf_threshold:.2f} confidence will be filtered OUT before SAM 2 sees them.")

        # GroundingDINO text-prompt filtering
        st.divider()
        st.markdown("**Text-Prompt Filtering (GroundingDINO)**")
        ground_status = get_grounding_status()
        if ground_status['available']:
            st.success(f"GroundingDINO loaded on {ground_status.get('device', 'unknown')}")
        else:
            if ground_status['error']:
                st.warning(f"GroundingDINO unavailable: {ground_status['error']}")
            else:
                st.caption("Install GroundingDINO to enable text filtering.")

        text_filter_enabled = st.checkbox(
            "Enable text-prompt filtering (GroundingDINO)",
            value=st.session_state.get("text_filter_enabled", False),
            help="Use GroundingDINO to detect regions matching text prompt. Works best for larger, well-defined objects.",
            key="text_filter_toggle",
            disabled=not ground_status['available']
        )
        st.session_state["text_filter_enabled"] = text_filter_enabled

        if text_filter_enabled:
            st.info("Text filtering is experimental. If GroundingDINO finds 0 regions, all YOLO boxes will be kept (fallback mode).")
        text_prompt = st.text_input(
            "Text prompt (e.g., 'stinkbug')",
            value=st.session_state.get("text_prompt", "stinkbug"),
            max_chars=80,
            disabled=not ground_status['available']
        )
        st.session_state["text_prompt"] = text_prompt

        iou_threshold = st.slider(
            "IoU threshold for text filter",
            min_value=0.1, max_value=0.9, value=float(st.session_state.get("iou_threshold", 0.5)), step=0.05,
            help="Minimum IoU between YOLO box and text-prompt box to keep detection.",
            disabled=not ground_status['available']
        )
        st.session_state["iou_threshold"] = iou_threshold

        if text_filter_enabled and ground_status['available']:
            st.caption("ℹLower IoU = stricter filtering (fewer boxes kept). Adjust if too many boxes are filtered out.")

    uploaded_files = st.file_uploader(
        "Upload RBSB images (multiple files allowed)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if uploaded_files:
        cache = st.session_state.get("detection_cache", {})
        new_files = [f for f in uploaded_files if f.name not in cache]

        if new_files:
            # Reset the submission flag when new files are uploaded
            st.session_state["report_saved_for_batch"] = False
            
            # Check if we should use SAM 2 verification (when approaching threshold)
            use_sam2 = st.session_state.get("use_sam2_verification", False)
            text_filter_enabled = st.session_state.get("text_filter_enabled", False)
            text_prompt = st.session_state.get("text_prompt", "")
            iou_threshold = st.session_state.get("iou_threshold", 0.5)
            yolo_conf_threshold = st.session_state.get("yolo_conf_threshold", 0.5)

            spinner_text = "Analyzing RBSB images with SAM 2 verification..." if use_sam2 else "Analyzing RBSB images. This may take a while depending upon the quality and number of images."

            with st.spinner(spinner_text):
                for file in new_files:
                    bytes_data = file.getvalue()
                    insect_count, annotated, metadata = count_insects(
                        bytes_data,
                        use_sam2_verification=use_sam2,
                        text_filter_enabled=text_filter_enabled,
                        text_prompt=text_prompt,
                        iou_threshold=iou_threshold,
                        yolo_conf_threshold=yolo_conf_threshold
                    )
                    cache[file.name] = {
                        "image_name": file.name,
                        "count": insect_count,
                        "preview": bytes_data,
                        "annotated": annotated,
                        "metadata": metadata,
                    }
            st.session_state["detection_cache"] = cache

            success_msg = "Detection complete (SAM 2 verified). Review results below." if use_sam2 else "Detection complete. Review results below."
            st.success(success_msg)

        # Build detections list from cache for all currently uploaded files
        detections = [cache[f.name] for f in uploaded_files if f.name in cache]
        st.session_state["latest_detections"] = detections

    detections = st.session_state.get("latest_detections", [])
    
    # Only show form and results if we have detections AND haven't already saved this batch
    if detections and not st.session_state.get("report_saved_for_batch", False):
        render_slideshow("Captured batch", detections, slider_key="capture_gallery")

        # Show annotation quality summary if SAM 2 was used
        sam2_used = any(det.get('metadata', {}).get('verified', False) for det in detections)
        if sam2_used:
            st.subheader("Annotation Quality Report")

            quality_data = []
            for det in detections:
                metadata = det.get('metadata', {})
                if metadata.get('verified'):
                    quality_data.append({
                        'Image': det['image_name'],
                        'YOLO Count': metadata.get('yolo_count', 0),
                        'Verified Count': metadata.get('yolo_verified_count', 0),
                        'Quality': metadata.get('annotation_quality', 'Unknown'),
                        'Issues': len(metadata.get('quality_issues', []))
                    })

            if quality_data:
                import pandas as pd
                quality_df = pd.DataFrame(quality_data)
                st.dataframe(quality_df, use_container_width=True)

                # Show overall statistics
                total_yolo = sum([d['YOLO Count'] for d in quality_data])
                total_verified = sum([d['Verified Count'] for d in quality_data])
                total_issues = sum([d['Issues'] for d in quality_data])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total YOLO Detections", total_yolo)
                with col2:
                    st.metric("High-Quality Detections", total_verified,
                             delta=f"{total_verified-total_yolo:+d}")
                with col3:
                    st.metric("Potential Issues Found", total_issues)

                if total_issues > 0:
                    st.warning(f"{total_issues} potential annotation issues detected. Review details below.")

        for det in detections:
            metadata = det.get('metadata', {})
            header = f"{det['image_name']} — {det['count']} insects"

            # Add quality indicator to header if verified
            if metadata.get('verified'):
                quality = ''
                quality_emoji = {'Excellent': '', 'Good': '✓', 'Fair': ' ', 'Poor': '❌'}.get(quality, '?')
                header += f" {quality_emoji} {quality}"

            with st.expander(header):
                st.image(det["annotated"], caption=f"Detected {det['count']} insects")
                st.image(det["preview"], caption="Original image")

                # Show text-filter details if applied
                if metadata.get('text_filter_applied'):
                    st.markdown("**Text-Prompt Filter Applied (Experimental):**")
                    st.write(f"- Prompt: '{metadata.get('text_prompt', 'N/A')}'")
                    st.write(f"- Original YOLO count: {metadata.get('yolo_original_count', 0)}")
                    st.write(f"- After confidence filter: {metadata.get('yolo_count_after_confidence_filter', 0)}")
                    st.write(f"- Final count: {metadata.get('yolo_count', 0)}")
                    
                    if metadata.get('grounding_dino_fallback'):
                        st.warning(f"roundingDINO found 0 '{metadata.get('text_prompt', '')}' regions. Kept all YOLO boxes (fallback mode). Text filtering may not work for small objects (<50px).")
                    elif metadata.get('grounding_dino_error'):
                        st.caption(f"GroundingDINO error: {metadata.get('grounding_dino_error', 'Unknown')}. Text filtering skipped.")
                    else:
                        st.write(f"- IoU threshold: {metadata.get('iou_threshold', 0):.2f}")
                        st.write(f"- Text boxes detected: {metadata.get('text_boxes_count', 0)}")

                # Show confidence filtering info
                st.markdown("**Pre-SAM2 Confidence Filter:**")
                conf_threshold = metadata.get('yolo_conf_threshold', 0.5)
                boxes_filtered = metadata.get('boxes_filtered_by_confidence', 0)
                if boxes_filtered > 0:
                    st.warning(f"Filtered out {boxes_filtered} low-confidence box(es) (< {conf_threshold:.2f}) BEFORE SAM 2. These likely were false positives (e.g., leaves).")
                else:
                    st.success(f"All {metadata.get('yolo_original_count', 0)} YOLO boxes passed confidence filter ({conf_threshold:.2f})")

                # Show verification details if available
                if metadata.get('verified'):
                    st.markdown("**SAM 2 Verification Details:**")
                    st.write(f"- YOLO detected: {metadata.get('yolo_count', 0)} insects")
                    st.write(f"- SAM 2 verified: {metadata.get('yolo_verified_count', 0)} high-quality")
                    st.write(f"- Annotation quality: **{metadata.get('annotation_quality', 'Unknown')}**")
                    st.write(f"- Average confidence: {metadata.get('avg_confidence', 0):.2f}")

                    # Show quality issues if any
                    quality_issues = metadata.get('quality_issues', [])
                    if quality_issues:
                        st.markdown("** Quality Issues Detected:**")
                        for issue in quality_issues:
                            box_idx = issue.get('box_index', '?')
                            issues_list = issue.get('issues', [])
                            st.write(f"  - Detection #{box_idx + 1}: {', '.join(issues_list)}")
                    else:
                        st.success("All detections passed quality checks")

        with st.form("location_form"):
            location = st.text_input("Location (city, block, GPS, etc.)", max_chars=120)
            submitted = st.form_submit_button("Save Report")

        if submitted and location.strip() and detections:
            # Prevent duplicate submission by checking if we already processed this batch
            if st.session_state.get("report_saved_for_batch") is True:
                st.warning("Report already submitted for this batch. Upload new images to submit again.")
                return
            
            # Mark this batch as submitted to prevent re-submission
            st.session_state["report_saved_for_batch"] = True
            
            user = st.session_state["auth_user"]
            insert_reports(user, location.strip(), detections)
            st.session_state["latest_detections"] = []
            st.session_state["detection_cache"] = {}
            
            # Update alerts and gallery
            threshold_check(location.strip(), user["email"])
            update_gallery(location.strip(), detections)
            
            # Clear caches to refresh dashboard
            try:
                get_reports_data.clear()
                get_location_stats_data.clear()
                get_gallery_data.clear()
            except Exception:
                pass
            
            st.success("Report saved successfully!")
            st.info("Upload new images to submit again.")
            st.rerun()  # Force rerun to hide the form immediately


def render_reports():
    st.subheader("Reports")
    df = load_reports()
    if df.empty:
        st.info("No reports yet. Capture and submit images to populate this table.")
        return

    # render_slideshow("Recent detections", load_gallery(), slider_key="reports_gallery")

    summary = (
        df.groupby("location")
        .agg(
            total_images=("image_name", "count"),
            total_insects=("insect_count", "sum"),
            latest_date=("report_date", "max"),
            priority=("priority", lambda x: "Priority" if "Priority" in set(x) else "Normal"),
        )
        .reset_index()
    )
    # summary["density_per_100_images"] = (
    #     summary.apply(
    #         lambda row: (row["total_insects"] / row["total_images"]) * 100 if row["total_images"] else 0, axis=1
    #     )
    #     .round(1)
    # )
    st.markdown("**Location overview (aggregated counts)**")
    st.dataframe(summary, use_container_width=True)
    st.caption("Applications recommended when density reaches 16 insects per 100 images in the same location.")

    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="stinkbug_reports.csv",
        mime="text/csv",
    )


def render_login_register():
    login_col, register_col = st.columns(2)

    with login_col:
        st.header("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Sign in"):
            user = authenticate_user(username.strip(), password)
            if user:
                st.session_state["auth_user"] = user
                st.session_state["menu_choice"] = "Homepage"
                st.success(f"Welcome back, {user['username']}! Redirecting to Homepage...")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with register_col:
        st.header("Register")
        new_username = st.text_input("New username", key="register_username")
        email = st.text_input("Email", key="register_email")
        new_password = st.text_input("New password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm password", type="password", key="register_confirm")
        if st.button("Create account"):
            if not all([new_username.strip(), email.strip(), new_password, confirm_password]):
                st.error("All fields are required.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                ok, msg = create_user(new_username.strip(), email.strip(), new_password)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

def get_ollama_client():
    """
    Initializes and returns an ollama client, handling potential connection issues.
    It attempts to connect to the Ollama server specified by the OLLAMA_SERVER_URL
    environment variable or defaults to http://localhost:11434.
    """
    ollama_server_url = os.getenv("OLLAMA_SERVER_URL", "http://localhost:11434")
    try:
        client = ollama.Client(host=ollama_server_url)
        client.list()
        print(f"Successfully connected to Ollama server at {ollama_server_url}")
        return client
    except ollama.ResponseError as e:
        print(f"Ollama Response Error: Could not connect to Ollama server at {ollama_server_url}. Error: {e}")
        return None
    except ConnectionRefusedError:
        print(f"Connection Refused Error: Ensure Ollama server is running at {ollama_server_url} and accessible.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while connecting to Ollama: {e}")
        return None

def get_reports_data():
    """
    Retrieves all stored reports as a list of dictionaries.
    """
    df = load_reports()
    return df.to_dict(orient='records')

def get_location_stats_data():
    """
    Retrieves statistics for each unique location, including total images, total insects, and density.
    """
    reports = get_reports_data()
    if not reports:
        return []

    locations = list(set([r['location'] for r in reports]))
    stats_data = []
    for location in locations:
        total_images, total_insects = location_stats(location)
        density_per_100_images = (total_insects / total_images) * 100 if total_images else 0
        stats_data.append({
            'location': location,
            'total_images': total_images,
            'total_insects': total_insects,
            'density_per_100_images': round(density_per_100_images, 1)
        })
    return stats_data

@st.cache_data
def get_gallery_data():
    """
    Retrieves recent gallery items, including raw image bytes.
    """
    gallery_items = load_gallery()
    formatted_items = []
    for item in gallery_items:
        formatted_items.append({
            'image_name': item['image_name'],
            'location': item['location'],
            'timestamp': item['timestamp'],
            'image_bytes': item['annotated'] # Include image bytes
        })
    return formatted_items

@st.cache_data
def get_notifications_data():
    """
    Retrieves the list of notifications from the Streamlit session state.
    """
    return st.session_state.get("notifications", [])

def generate_ollama_prompt(user_question: str) -> str:
    """
    Generates a contextual prompt for the Ollama model based on the user's question
    and retrieved application data. Uses a grounding technique to ensure responses
    are based strictly on the provided context.
    """
    reports_data = get_reports_data()
    location_stats_data = get_location_stats_data()
    gallery_data = get_gallery_data()
    notifications_data = get_notifications_data()

    def normalize_loc(loc: str) -> str:
        import re
        loc = loc or ""
        loc = loc.split(",")[0]
        return re.sub(r"[^a-z0-9]", "", loc.casefold())

    context_str = ""

    context_str += "## REPORTS DATA (Stink Bug Detection Records):\n"
    if reports_data:
        context_str += "| Username | Date | Location | Image Name | Bug Count | Priority |\n"
        context_str += "|----------|------|----------|------------|-----------|----------|\n"
        for report in reports_data:
            # Apply 0→1 correction for display
            insect_count = report['insect_count']
            if insect_count == 0 and report.get('image_name'):
                insect_count = 1
            context_str += f"| {report['username']} | {report['report_date']} | {report['location']} | {report['image_name']} | {insect_count} | {report['priority']} |\n"
    else:
        context_str += "No reports available.\n"

    context_str += "\n## LOCATION STATISTICS (Summary by Location):\n"
    if location_stats_data:
        context_str += "| Location | Total Images | Total Insects | Density (per 100) |\n"
        context_str += "|----------|--------------|---------------|-------------------|\n"
        for stats in location_stats_data:
            context_str += f"| {stats['location']} | {stats['total_images']} | {stats['total_insects']} | {stats['density_per_100_images']} |\n"
    else:
        context_str += "No location statistics available.\n"

    context_str += "\n## AVAILABLE IMAGES (Gallery):\n"
    if gallery_data:
        # Build normalized map: loc_key -> {display_name, images[]}
        images_by_loc = {}
        for item in gallery_data:
            loc = item.get('location', '') or "Unknown"
            key = normalize_loc(loc)
            if key not in images_by_loc:
                images_by_loc[key] = {"display": loc, "images": []}
            images_by_loc[key]["images"].append(item)

        available_image_names = []
        for key, bundle in images_by_loc.items():
            display = bundle["display"] or "Unknown"
            context_str += f"- Location: {display} (key: {key})\n"
            for item in bundle["images"]:
                image_name = item['image_name']
                available_image_names.append(image_name)
                has_stink_bugs = any(r['image_name'] == image_name and r['insect_count'] > 0 for r in reports_data)
                image_status = "✓ Contains Stink Bugs" if has_stink_bugs else "✗ No/Few Bugs"
                context_str += f"    • {image_name} | {image_status} | Captured: {item['timestamp']}\n"
        if available_image_names:
            context_str += f"\n### Image names available for display:\n{', '.join(available_image_names)}\n"
    else:
        context_str += "No gallery images available.\n"

    context_str += "\n## RECENT ALERTS/NOTIFICATIONS:\n"
    if notifications_data:
        for notification in notifications_data:
            context_str += f"- [{notification['timestamp']}] {notification['message']} (Location: {notification['location']}, Ratio: {notification['ratio']})\n"
    else:
        context_str += "No alerts or notifications.\n"

    # Explicit model stack so the LLM can answer stack questions without guessing
    context_str += "\n## MODEL STACK:\n"
    context_str += "- Detector: YOLOv8m with CBAM + ASFF\n"
    context_str += "- Verification: SAM2 (optional)\n"
    context_str += "- Text-guided filtering: GroundingDINO (optional)\n"
    context_str += "- Chat/LLM: Ollama with Mistral (primary) and Llama2 (fallback)\n"

    # Retrieved facts (grounding) for numeric answers
    if location_stats_data:
        # Sort by density desc for quick reference
        sorted_by_density = sorted(location_stats_data, key=lambda x: x['density_per_100_images'], reverse=True)
        top_density = sorted_by_density[:5]
        low_detection = [s for s in location_stats_data if s['total_images'] > 0 and (s['total_insects'] / s['total_images']) < 0.5]

        context_str += "\n## RETRIEVED FACTS (ground truth for answers):\n"
        context_str += "Top density locations (per 100 images, and per image):\n"
        for s in top_density:
            per_img = round(s['total_insects'] / s['total_images'], 2) if s['total_images'] else 0
            context_str += f"- {s['location']}: insects={s['total_insects']}, images={s['total_images']}, density={s['density_per_100_images']}/100, per_image={per_img}\n"

        if low_detection:
            context_str += "Low detection rate locations (<0.5 insects per image):\n"
            for s in low_detection[:5]:
                per_img = round(s['total_insects'] / s['total_images'], 2) if s['total_images'] else 0
                context_str += f"- {s['location']}: insects={s['total_insects']}, images={s['total_images']}, per_image={per_img}\n"

    # Add data quality analysis section - use CORRECTED stats, not raw reports_data
    context_str += "\n## DATA QUALITY ANALYSIS (for smart recommendations):\n"
    if location_stats_data:
        # Check for anomalies using corrected location stats
        zero_or_low_locs = [s for s in location_stats_data if s['total_insects'] < s['total_images'] * 0.5]
        if zero_or_low_locs:
            context_str += f"LOW DETECTION RATE: {len(zero_or_low_locs)} locations with <0.5 insects per image:\n"
            for s in zero_or_low_locs[:3]:
                context_str += f"   • {s['location']}: {s['total_insects']} insects / {s['total_images']} images = {s['total_insects']/s['total_images']:.2f} per image\n"
                context_str += f"     → Recommendation: Review image quality OR lower confidence threshold OR enable SAM2\n"
        
        # Density outliers
        densities = [s['density_per_100_images'] for s in location_stats_data]
        if densities:
            avg_density = sum(densities) / len(densities)
            high_density_locs = [s for s in location_stats_data if s['density_per_100_images'] > avg_density * 2]
            if high_density_locs:
                context_str += f"HIGH DENSITY LOCATIONS (>{avg_density*2:.1f}/100, avg: {avg_density:.1f}/100):\n"
                for s in high_density_locs:
                    context_str += f"   • {s['location']}: {s['density_per_100_images']}/100 - Consider SAM2 verification or stricter confidence threshold\n"
        
        context_str += "\n"
    
    prompt = f"""SYSTEM INSTRUCTION: You are a specialized AI assistant for a Redbanded Stink Bug detection application. 
YOUR RESPONSES MUST BE BASED ENTIRELY ON THE PROVIDED CONTEXT DATA BELOW.
DO NOT USE EXTERNAL KNOWLEDGE, DO NOT MAKE UP INFORMATION, AND DO NOT HALLUCINATE DETAILS.

EXCEPTION FOR REASONING REQUESTS: If the user asks you to explain reasons, causes, effects, consequences, or provide analysis (e.g., "why", "what could be the reason", "what could be the effect"), you MAY use domain knowledge about stink bugs, environmental factors, and agricultural science to provide thoughtful analysis ALONGSIDE the data-driven facts.

If the user's question is purely factual and cannot be answered using ONLY the information in the provided context, you MUST state: "I cannot answer this question with the available information."

=== START OF CONTEXT DATA ===

{context_str}

=== END OF CONTEXT DATA ===

INSTRUCTIONS FOR DISPLAYING IMAGES:
- When the user asks for images, IMMEDIATELY display them using [DISPLAY_IMAGE:filename] format
- DO NOT ask "would you like to see" or "let me know if you want" - JUST SHOW THEM
- DO NOT list filenames as text - USE THE TAG FORMAT: [DISPLAY_IMAGE:filename]
- Only use image names from the "Image names available for display" list above
- The filename MUST match EXACTLY (including extension)
- Example: [DISPLAY_IMAGE:AdltStnkBug_005_3500_jpg.rf.5c35d5066f8d8fe6b860e5844d50631d.jpg]
- If user says "yes", "show me", "I want to see" after you mentioned images, display them immediately with tags
- IMPORTANT: Background/environment metadata (green leaf, cardboard, plant, etc.) is NOT tracked in the system
- If user requests images with specific background filters (e.g., "on green leaf", "on plant", "on cardboard"):
  * FIRST state clearly: "Background/environment data is not available in the system."
  * THEN show available stinkbug images anyway
  * Suggest they can visually inspect the images to find what they need

RULES FOR ANSWERING:
1. For FACTUAL questions: ONLY reference data present in the context above
2. For REASONING/ANALYSIS questions: Ground answers in the data, but expand with domain knowledge to explain "why" and "what could be the effect"
3. When asked about insects/bugs, use the exact numbers from REPORTS DATA or LOCATION STATISTICS
4. When asked about locations, reference LOCATION STATISTICS section
5. When asked about priorities, use the exact Priority values from REPORTS DATA
6. When asked about dates/timestamps, use exact values from the data
7. Do NOT infer, estimate, or calculate values not present in the data
8. Keep your response concise and directly address the question
9. If asked about something not in the data, explicitly state it's not available
10. For image requests: IMMEDIATELY show images using [DISPLAY_IMAGE:filename] tags. DO NOT ask for confirmation. Show up to 6 images. Parse locations case-insensitively using the location key from AVAILABLE IMAGES.
11. If user says "yes", "show", "display" after discussing images, display them NOW with [DISPLAY_IMAGE:filename] tags
12. For model/stack questions: Answer exactly from MODEL STACK. Do NOT say "not provided"; the stack is listed.
13. For numbers (counts, densities, rankings): Use RETRIEVED FACTS as ground truth. Do NOT invent values.
14. When explaining REASONS/EFFECTS: You may reference domain knowledge about stink bug biology, agricultural practices, and environmental factors

DATA-DRIVEN DECISION SUPPORT (Smart Analysis Mode):
When user asks for:
- "recommendations", "improve accuracy", "optimize", "suggestions", "analysis", "insights"
- "what's wrong with", "anomalies", "issues", "problems"
- "should I enable", "which settings", "best threshold"

Provide ACTIONABLE recommendations based on DATA QUALITY ANALYSIS section:
- If 0-count images exist → Recommend lowering YOLO confidence threshold OR enabling SAM2 verification
- If high-density locations detected → Recommend SAM2 verification + GroundingDINO filtering for those locations
- If density varies widely → Suggest location-specific threshold tuning
- If consistent detection failures → Recommend reviewing image quality or model retraining
- Compare location performance and suggest best practices from high-performing locations

Example smart responses:
Q: "How can I improve counting accuracy?"
A: "Based on analysis: Arizona shows 0 insects in 1 uploaded image—possible detection failure. Recommendation: Lower YOLO confidence threshold from 0.5 to 0.4 OR enable SAM2 verification to catch missed detections."

Q: "Should I adjust settings for high-density areas?"
A: "Yes. Butwal shows density of 700/100 (7x average). Recommendation: Enable GroundingDINO text filtering ('stinkbug') to reduce false positives from leaves/debris in crowded images."

REASONING AND ANALYSIS REQUESTS:
When user asks WHY, REASON, EFFECT, CONSEQUENCE, or requests analysis/explanation:
1. FIRST: State the factual data (e.g., location X has Y insects, location Z has W insects)
2. THEN: Provide reasoning using domain knowledge about stink bugs:
   - Geographic/climate factors affecting stink bug populations
   - Seasonal variations (temperature, humidity, crop cycles)
   - Agricultural practices in different locations
   - Habitat suitability and food sources
   - Environmental conditions that favor stink bug reproduction
3. EFFECTS/CONSEQUENCES: Discuss agricultural impact, economic implications, crop damage patterns
4. Explicitly connect the reasoning to the DATA you see in this application

Example reasoning response:
Q: "Which location has the most stinkbugs? Why might that be?"
A: "According to the data, Location A has 250 stinkbugs across 50 images (5 per image average), while Location B has only 30 stinkbugs across 40 images (0.75 per image).

The higher population in Location A could be due to: [domain knowledge about why that location might be favorable]. This aligns with observed patterns where [environmental/agricultural factors] tend to support larger stink bug populations."

USER QUESTION: {user_question}

RESPONSE (answer based ONLY on the context data provided above):"""
    return prompt


def validate_response_grounding(ai_response: str, user_question: str, context_data: dict) -> tuple[bool, str]:
    """
    Validates if the AI response appears to be grounded in the provided context.
    Returns (is_grounded, warning_message).
    """
    # Check if response contains explicit "cannot answer" statement
    cannot_answer_phrases = [
        "cannot answer",
        "not available",
        "no information",
        "not found in",
        "not present in the context"
    ]
    
    if any(phrase.lower() in ai_response.lower() for phrase in cannot_answer_phrases):
        return True, ""  # Valid - explicitly states lack of information
    
    # Check if response references actual data from context
    reports_data = context_data.get('reports_data', [])
    location_stats = context_data.get('location_stats', [])
    gallery_items = context_data.get('gallery_items', [])
    
    # Extract specific data points from context
    locations = [s.get('location') for s in location_stats if s.get('location')]
    image_names = [g.get('image_name') for g in gallery_items if g.get('image_name')]
    usernames = [r.get('username') for r in reports_data if r.get('username')]
    
    # Check if response contains verifiable references to context
    response_lower = ai_response.lower()
    context_references = []
    
    # Count references to locations
    for location in locations:
        if location and location.lower() in response_lower:
            context_references.append('location')
            break
    
    # Count references to specific numbers from reports
    for report in reports_data:
        if str(report.get('insect_count', '')) in response_lower:
            context_references.append('number')
            break
    
    # Count references to image names
    for image_name in image_names:
        if image_name and image_name.lower() in response_lower:
            context_references.append('image')
            break
    
    # Check for obvious hallucinations
    if "i'm not sure" in response_lower and len(context_references) == 0:
        return False, "  Response may not be fully grounded in available data. Please verify the information."
    
    if len(context_references) == 0 and len(reports_data) > 0 and len(location_stats) > 0:
        # Has context but no references - might be hallucinating
        return False, "  Response may not reference the available data. Please verify the information."
    
    return True, ""


def fast_answer(user_question: str) -> str | None:
    """Returns an instant, deterministic answer for common questions without calling the model."""
    q = user_question.lower().strip()
    
    # Check if question requires reasoning/analysis - skip fast answer
    requires_reasoning = any(word in q for word in [
        "why", "reason", "explain", "understand", "effect", "impact", "consequence",
        "cause", "because", "analysis", "analyze", "interpret", "mean", "suggest",
        "recommend", "advise", "solution", "help me", "what could be"
    ])
    
    if requires_reasoning:
        return None  # Let LLM handle it with proper reasoning
    
    # Check if this is a smart analysis request (trigger LLM for recommendations)
    analysis_keywords = ["recommend", "improve", "accuracy", "optimize", "suggest", "analysis", "insight", 
                        "anomal", "issue", "problem", "wrong", "should i enable", "which setting", 
                        "best threshold", "how to improve", "what's wrong", "detection failure"]
    if any(keyword in q for keyword in analysis_keywords):
        return None  # Let LLM handle smart analysis with full context
    
    # Get fresh reports data (not cached) for accurate location counts
    reports = get_reports_data()
    if not reports:
        return None
    
    # Build location statistics from fresh reports (same as chart)
    import re
    from collections import defaultdict
    
    location_data = {}
    for report in reports:
        loc = report.get('location', '').strip()
        if loc:
            if loc not in location_data:
                location_data[loc] = {'images': 0, 'insects': 0}
            location_data[loc]['images'] += 1
            # Count insects: use stored count, but if 0 and image exists, count as at least 1
            insect_count = report.get('insect_count', 0)
            if insect_count == 0 and report.get('image_name'):
                insect_count = 1
            location_data[loc]['insects'] += insect_count
    
    # Normalize locations same as chart does
    def normalize_loc(loc: str) -> str:
        loc = loc or ""
        loc = loc.split(",")[0]  # collapse variants like "Butwal, Nepal" vs "Butwal"
        return re.sub(r"[^a-z0-9]", "", loc.casefold())
    
    location_reports = defaultdict(list)
    location_strings = {}
    for loc, data in location_data.items():
        key = normalize_loc(loc)
        location_reports[key].append({
            'insects': data['insects'],
            'images': data['images']
        })
        if key not in location_strings:
            location_strings[key] = loc
        else:
            if len(loc) > len(location_strings[key]):
                location_strings[key] = loc
    
    # Build normalized location_counts
    location_counts = {}
    for key, reports_list in location_reports.items():
        total_insects = sum(r['insects'] for r in reports_list)
        total_images = sum(r['images'] for r in reports_list)
        location_counts[location_strings.get(key, key)] = {
            'total_insects': total_insects,
            'total_images': total_images,
            'density_per_100_images': round((total_insects / total_images) * 100, 1) if total_images > 0 else 0
        }

    # Check if user is asking for images from a specific location - don't respond, let fast_image_response handle it
    if "image" in q or "photo" in q or "picture" in q or "show me" in q or "display" in q or "give me" in q:
        return None  # Let fast_image_response handle this
    
    # Total statistics (overall queries)
    total_insects_all = sum(loc['total_insects'] for loc in location_counts.values())
    total_images_all = sum(loc['total_images'] for loc in location_counts.values())
    
    # Location list queries - "how many cities", "what are the cities", "list locations"
    if any(word in q for word in ["how many", "what are", "list", "which"]):
        if any(word in q for word in ["location", "city", "cities", "area", "areas", "place", "places", "region"]):
            if any(word in q for word in ["are there", "are in", "do we have", "recorded", "record", "available", "what are"]):
                return f"We have data from {len(location_counts)} locations: {', '.join(location_counts.keys())}."
    
    # Overall/total queries
    if any(word in q for word in ["total", "overall", "all", "entire"]):
        if any(word in q for word in ["how many", "count", "number"]):
            if "location" in q or "city" in q or "cities" in q or "area" in q:
                return f"Data collected from {len(location_counts)} locations: {', '.join(location_counts.keys())}."
            return f"Total: {total_insects_all} stinkbugs detected across {total_images_all} images."
    
    # Summary/status queries
    if any(word in q for word in ["summary", "overview", "status", "report", "statistics", "stats"]):
        top_loc = max(location_counts.items(), key=lambda x: x[1]['total_insects']) if location_counts else None
        if top_loc:
            return f"Summary: {total_insects_all} stinkbugs in {total_images_all} images across {len(location_counts)} locations. Highest: {top_loc[0]} ({top_loc[1]['total_insects']} stinkbugs)."
    
    # Check if user is asking about counts/insects in a specific location
    if any(word in q for word in ["how many", "count", "total", "number of", "insects", "stinkbugs", "bugs", "present", "found", "detected"]):
        # Normalize question
        q_normalized = normalize_loc(q)
        q_clean = q.replace("?", "").replace(".", "").replace(",", "").replace("!", "").lower()
        
        # Try to find matching location in question (case-insensitive)
        for loc_name, loc_data in location_counts.items():
            loc_normalized = normalize_loc(loc_name)
            # Check if location matches (normalized)
            if loc_normalized in q_normalized or loc_normalized in q_clean:
                total_insects = loc_data['total_insects']
                total_images = loc_data['total_images']
                density = loc_data['density_per_100_images']
                #return f"In {loc_name}: {total_insects} stinkbug(s) detected across {total_images} image(s)."
                return f"We found {total_insects} stinkbug(s) in {total_images} image(s) from {loc_name}."
    
    # For highest/lowest queries, use the pre-computed cached data
    location_stats = get_location_stats_data()
    
    # Highest-location question (fast path)
    if "highest" in q and ("city" in q or "location" in q or "area" in q or "region" in q or "place" in q):
        best = max(location_counts.items(), key=lambda x: x[1]['total_insects'])
        if best:
            return f"{best[0]} has the highest redbanded stink bug count ({best[1]['total_insects']})."
    
    # Most/maximum queries
    if any(word in q for word in ["most", "maximum", "max", "top"]):
        if any(term in q for term in ["city", "location", "area", "region", "place", "stinkbugs", "bugs", "insects"]):
            best = max(location_counts.items(), key=lambda x: x[1]['total_insects'])
            if best:
                return f"{best[0]} has the most stinkbugs ({best[1]['total_insects']})."

    # Lowest-location question (fast path)
    if any(word in q for word in ["lowest", "least", "minimum", "fewest", "smallest"]):
        if any(term in q for term in ["city", "location", "area", "region", "place", "stinkbugs", "bugs", "insects"]):
            worst = min(location_counts.items(), key=lambda x: x[1]['total_insects'])
            if worst:
                return f"{worst[0]} has the fewest stinkbugs ({worst[1]['total_insects']})."
    
    # Comparison between two locations
    if " vs " in q or " versus " in q or "compare" in q or "difference between" in q:
        q_normalized = normalize_loc(q)
        matched_locs = []
        for loc_name in location_counts.keys():
            if normalize_loc(loc_name) in q_normalized:
                matched_locs.append(loc_name)
        
        if len(matched_locs) >= 2:
            loc1, loc2 = matched_locs[0], matched_locs[1]
            count1 = location_counts[loc1]['total_insects']
            count2 = location_counts[loc2]['total_insects']
            diff = abs(count1 - count2)
            return f"{loc1}: {count1} stinkbugs, {loc2}: {count2} stinkbugs. Difference: {diff}."

    return None


def _parse_report_date(report_date_str: str) -> datetime | None:
    """Parses stored report_date strings like '01::25::26' or '01-25-2025'."""
    if not report_date_str:
        return None
    normalized = report_date_str.replace("::", "-").replace("/", "-")
    fmts = ["%m-%d-%y", "%m-%d-%Y", "%m-%d-%y", "%m-%d-%Y"]
    for fmt in fmts:
        try:
            return datetime.strptime(normalized, fmt)
        except Exception:
            continue
    return None


def _parse_user_date_range(user_question: str) -> tuple[datetime | None, datetime | None]:
    """Extracts an optional start/end datetime from the question (month/year, specific date, or between two dates)."""
    text = user_question.lower()
    # between X and Y pattern
    between_match = re.search(r"between\s+([^\s].*?)\s+and\s+([^\s].*?)($|\.|,)", text)
    if between_match:
        d1 = between_match.group(1).strip()
        d2 = between_match.group(2).strip()
        start = _parse_human_date(d1)
        end = _parse_human_date(d2)
        if start and end:
            return start, end
    
    # single specific date pattern like "11:18:2025" or "11-18-2025" or "11/18/2025"
    date_match = re.search(r"(\d{1,2})[:\/-](\d{1,2})[:\/-](\d{2,4})", text)
    if date_match:
        m, d, y = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
        if y < 100:
            y += 2000
        try:
            specific_date = datetime(y, m, d)
            # Return full day: start at midnight, end at 23:59:59
            end_of_day = specific_date + timedelta(days=1) - timedelta(seconds=1)
            return specific_date, end_of_day
        except Exception:
            pass
    
    # single month/year pattern like "january 2025" or "october, 2025"
    month_year_match = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s*,?\s*(\d{2,4})", text)
    if month_year_match:
        month = month_year_match.group(1)
        year = month_year_match.group(2)
        month_num = ["january","february","march","april","may","june","july","august","september","october","november","december"].index(month) + 1
        year_num = int(year)
        if year_num < 100:
            year_num += 2000
        start = datetime(year_num, month_num, 1)
        # month end: next month minus one day
        if month_num == 12:
            end = datetime(year_num + 1, 1, 1) - timedelta(days=1)
        else:
            end = datetime(year_num, month_num + 1, 1) - timedelta(days=1)
        return start, end
    return None, None


def _parse_human_date(text: str) -> datetime | None:
    """Parses human-friendly dates like 'october 2025', '01-25-2025', '01/25/25'."""
    text = text.strip().replace("/", "-")
    # month name + year
    m = re.match(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{2,4})", text)
    if m:
        month = m.group(1)
        year = int(m.group(2))
        if year < 100:
            year += 2000
        month_num = ["january","february","march","april","may","june","july","august","september","october","november","december"].index(month) + 1
        return datetime(year, month_num, 1)
    # numeric formats
    fmts = ["%m-%d-%Y", "%m-%d-%y", "%m-%Y", "%m-%y"]
    for fmt in fmts:
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue
    return None


def fast_image_response(user_question: str):
    """Returns deterministic image response(s) for location-based image requests without calling the model."""
    q = user_question.lower()
    
    # Check if question requires reasoning/analysis - skip fast image response
    requires_reasoning = any(word in q for word in [
        "why", "reason", "explain", "understand", "effect", "impact", "consequence",
        "cause", "because", "analysis", "analyze", "interpret", "mean", "suggest",
        "recommend", "advise", "solution", "help me", "highest", "lowest", "compare"
    ])
    
    if requires_reasoning:
        return None  # Let LLM handle it
    
    if "image" not in q and "photo" not in q and "picture" not in q:
        return None
    
    # Check for background/environment filter requests that we can't fulfill
    background_keywords = ["green leaf", "leaf", "plant", "cardboard", "background", "surface", "wood", "paper", "environment"]
    has_background_filter = any(keyword in q for keyword in background_keywords)
    
    if has_background_filter:
        # Return a message explaining limitation + show all available images anyway
        reports = get_reports_data()
        if not reports:
            return None
        
        gallery = {item['image_name']: item for item in get_gallery_data()}
        all_reports = []
        for report in reports:
            if report.get('image_name'):
                rdate = _parse_report_date(report.get('report_date', ''))
                all_reports.append((rdate or datetime.min, report))
        
        all_reports.sort(key=lambda x: x[0], reverse=True)
        
        image_list = []
        for _, report in all_reports[:6]:  # Show up to 6 images
            image_name = report['image_name']
            gallery_item = gallery.get(image_name)
            if gallery_item and gallery_item.get('image_bytes'):
                caption = f"Image: {image_name} from {report.get('location', 'Unknown')}"
                if report.get('report_date'):
                    caption += f" | {report['report_date']}"
                if report.get('insect_count') is not None:
                    caption += f" | Insects: {report['insect_count']}"
                image_list.append({
                    "location": report.get('location', 'Various'),
                    "image_name": image_name,
                    "image_bytes": gallery_item['image_bytes'],
                    "caption": caption,
                })
        
        if image_list:
            return {
                "location": "  Note: Background/environment filtering not available",
                "images": image_list,
                "is_plural": True,
                "filter_note": "Background metadata (leaf color, surface type) is not tracked. Please visually inspect these images to find what you need."
            }

    reports = get_reports_data()
    if not reports:
        return None

    # Build lookup tables
    gallery = {item['image_name']: item for item in get_gallery_data()}
    locations = {r['location'] for r in reports if r.get('location')}

    # Date filters
    start_date, end_date = _parse_user_date_range(user_question)

    # Normalize location function (same as used in fast_answer)
    import re
    def normalize_loc(loc: str) -> str:
        loc = loc or ""
        loc = loc.split(",")[0]  # collapse variants like "Butwal, Nepal" vs "Butwal"
        return re.sub(r"[^a-z0-9]", "", loc.casefold())

    # Find a location mentioned in the question
    target_location = None
    q_normalized = normalize_loc(q)
    q_words = set(q.lower().split())
    
    # Try exact normalized matching first
    for loc in locations:
        if loc:
            loc_normalized = normalize_loc(loc)
            # Check if normalized location appears in normalized question
            if loc_normalized and loc_normalized in q_normalized:
                target_location = loc
                break
    
    # If not found, try matching individual location words
    if not target_location:
        for loc in locations:
            if loc:
                # Check if any word from location appears in question
                loc_words = set(loc.lower().replace(',', ' ').split())
                if loc_words & q_words:  # Intersection of sets
                    target_location = loc
                    break

    # If no specific location mentioned, check if it's a generic request for any images
    if not target_location:
        # Generic requests like "show me sample images", "display stinkbug images"
        if any(word in q for word in ["sample", "any", "some", "random", "example"]):
            # Get images from any location (up to 6)
            all_reports = []
            for report in reports:
                if report.get('image_name'):
                    rdate = _parse_report_date(report.get('report_date', ''))
                    all_reports.append((rdate or datetime.min, report))
            
            all_reports.sort(key=lambda x: x[0], reverse=True)
            
            image_list = []
            for _, report in all_reports[:6]:  # Limit to 6 images
                image_name = report['image_name']
                gallery_item = gallery.get(image_name)
                if gallery_item and gallery_item.get('image_bytes'):
                    caption = f"Image: {image_name} from {report.get('location', 'Unknown')}"
                    if report.get('report_date'):
                        caption += f" at {report['report_date']}"
                    if report.get('insect_count') is not None:
                        caption += f" | Insects: {report['insect_count']}"
                    image_list.append({
                        "location": report.get('location', 'Various'),
                        "image_name": image_name,
                        "image_bytes": gallery_item['image_bytes'],
                        "caption": caption,
                    })
            
            if image_list:
                return {
                    "location": "Various locations",
                    "images": image_list,
                    "is_plural": True
                }
        
        return None

    # Check if asking for multiple images (plural "images" or "sample")
    is_plural = "images" in q or "photos" in q or "pictures" in q or "sample" in q

    # Find all location variants that match the target (case-insensitive)
    target_normalized = normalize_loc(target_location)
    matching_locations = [loc for loc in locations if normalize_loc(loc) == target_normalized]

    # Filter reports for that location (and its variants) within date range
    filtered = []
    for report in reports:
        report_loc = report.get('location')
        if not report_loc or normalize_loc(report_loc) != target_normalized:
            continue
        if not report.get('image_name'):
            continue
        rdate = _parse_report_date(report.get('report_date', ''))
        if start_date and rdate and rdate < start_date:
            continue
        if end_date and rdate and rdate > end_date:
            continue
        filtered.append((rdate or datetime.min, report))

    # Sort newest first
    filtered.sort(key=lambda x: x[0], reverse=True)

    # If asking for multiple images, return all; otherwise return just one
    if is_plural:
        image_list = []
        for _, report in filtered:
            image_name = report['image_name']
            gallery_item = gallery.get(image_name)
            if gallery_item and gallery_item.get('image_bytes'):
                caption = f"Image: {image_name} from {target_location}"
                if report.get('report_date'):
                    caption += f" at {report['report_date']}"
                if report.get('insect_count') is not None:
                    caption += f" | Insects: {report['insect_count']}"
                image_list.append({
                    "location": target_location,
                    "image_name": image_name,
                    "image_bytes": gallery_item['image_bytes'],
                    "caption": caption,
                })
        
        if image_list:
            return {
                "location": target_location,
                "images": image_list,  # Multiple images
                "is_plural": True
            }
    else:
        # Return single image
        for _, report in filtered:
            image_name = report['image_name']
            gallery_item = gallery.get(image_name)
            if gallery_item and gallery_item.get('image_bytes'):
                caption = f"Image: {image_name} from {target_location}"
                if report.get('report_date'):
                    caption += f" at {report['report_date']}"
                if report.get('insect_count') is not None:
                    caption += f" | Insects: {report['insect_count']}"
                return {
                    "location": target_location,
                    "image_name": image_name,
                    "image_bytes": gallery_item['image_bytes'],
                    "caption": caption,
                }
    
    return None

    return None


@st.cache_data
def get_response_context() -> dict:
    """
    Retrieves context data for response validation.
    """
    return {
        'reports_data': get_reports_data(),
        'location_stats': get_location_stats_data(),
        'gallery_items': get_gallery_data(),
    }


def save_feedback_to_db(user_query: str, assistant_response: str, label: str, response_type: str):
    """Persist feedback to database for learning and analytics."""
    from datetime import datetime
    try:
        username = st.session_state.get("auth_user", "unknown")
        if isinstance(username, dict):
            username = username.get("username", "unknown")
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO feedback (user_query, assistant_response, label, response_type, created_at, username) VALUES (?, ?, ?, ?, ?, ?)",
                (user_query, assistant_response, label, response_type, datetime.now().isoformat(), username)
            )
            conn.commit()
            print(f"Feedback saved: {label} for response_type={response_type}")  # Debug
    except Exception as e:
        print(f"Error saving feedback: {e}")
        st.error(f"Could not save feedback: {e}")


def try_database_answer(user_query: str) -> Optional[Dict]:
    """
    SEMANTIC UNDERSTANDING FIRST, then query database.
    
    Process:
    1. Use LLM to understand semantic intent (or regex fallback if Ollama unavailable)
    2. Extract structured query parameters
    3. Query actual database based on understanding
    4. Return validated results from OUR data
    
    NOTE: If query requires reasoning/analysis/explanation, return None to let LLM handle it
    """
    
    # Check if query requires reasoning/analysis beyond just data lookup
    query_lower = user_query.lower()
    requires_reasoning = any(word in query_lower for word in [
        "why", "reason", "explain", "understand", "effect", "impact", "consequence",
        "cause", "because", "analysis", "analyze", "interpret", "mean", "suggest",
        "recommend", "advise", "solution", "help me"
    ])
    
    # If multi-part complex question requiring reasoning, let LLM handle it
    if requires_reasoning:
        # Return None to fall through to Agent/LLM which can provide reasoning
        return None
    
    # Step 1: Semantic Understanding using LLM (or regex fallback)
    intent = None
    
    if st.session_state.get("ollama_client"):
        # Use LLM for understanding
        understanding_prompt = f"""Analyze this user question and extract the intent in JSON format.

User Question: {user_query}

Determine:
1. query_type: "comparison" (finding highest/lowest/most/least), "specific_location" (stats for one place), "trend" (over time), "count" (how many total), or "other"
2. comparison_type: "highest", "lowest", "compare_all", or null
3. metric: "total_count" (total insects), "density" (per 100 images), "image_count", or null
4. location: specific location name if mentioned, or null
5. time_period: number of days if mentioned, or null

Return ONLY valid JSON, no other text:
{{"query_type": "...", "comparison_type": "...", "metric": "...", "location": "...", "time_period": null}}

JSON:"""

        try:
            response = st.session_state["ollama_client"].chat(
                model='mistral',
                messages=[{'role': 'user', 'content': understanding_prompt}],
                options={"temperature": 0.1, "num_predict": 150}
            )
            
            # Parse LLM understanding
            understanding_text = response['message']['content'].strip()
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', understanding_text, re.DOTALL)
            if json_match:
                intent = json.loads(json_match.group())
                print(f"🧠 Semantic Understanding (LLM): {intent}")  # Debug
        except Exception as e:
            print(f"Semantic understanding failed: {e}")
            intent = None
    
    # If LLM unavailable or failed, use regex-based fallback
    if not intent:
        query_lower = user_query.lower()
        
        # Detect query type from patterns
        if any(word in query_lower for word in ["how many", "how much", "count", "total", "number of", "many"]):
            if any(word in query_lower for word in ["cities", "locations", "places", "areas"]):
                intent = {
                    "query_type": "count",
                    "comparison_type": None,
                    "metric": "locations",
                    "location": None
                }
            else:
                intent = {
                    "query_type": "count",
                    "comparison_type": None,
                    "metric": "total_count",
                    "location": None
                }
        elif any(word in query_lower for word in ["highest", "most", "maximum", "greatest"]):
            intent = {
                "query_type": "comparison",
                "comparison_type": "highest",
                "metric": "total_count",
                "location": None
            }
        elif any(word in query_lower for word in ["lowest", "least", "minimum", "fewest"]):
            intent = {
                "query_type": "comparison",
                "comparison_type": "lowest",
                "metric": "total_count",
                "location": None
            }
        elif any(word in query_lower for word in ["trend", "trend analysis"]):
            intent = {
                "query_type": "trend",
                "comparison_type": None,
                "metric": None,
                "location": None
            }
        
        if intent:
            print(f"🧠 Semantic Understanding (Regex): {intent}")  # Debug
    
    if not intent:
        return None
    
    # Step 2: Query database based on semantic understanding
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Handle comparison queries
            if intent.get("query_type") == "comparison":
                # Query all location data
                cursor.execute("""
                    SELECT location, COUNT(*) as image_count, SUM(insect_count) as total_insects
                    FROM reports 
                    GROUP BY location
                    ORDER BY total_insects DESC
                """)
                rows = cursor.fetchall()
                
                if not rows:
                    return None
                
                # Calculate stats for each location
                all_stats = []
                for loc, img_count, total_insects in rows:
                    total_insects = total_insects or 0
                    density = (total_insects / img_count * 100) if img_count > 0 else 0
                    all_stats.append({
                        'location': loc,
                        'image_count': img_count,
                        'total_insects': total_insects,
                        'density': round(density, 2)
                    })
                
                # Determine metric to use
                if intent.get("metric") == "density":
                    metric_key = 'density'
                    metric_label = "density"
                    metric_unit = "per 100 images"
                else:
                    metric_key = 'total_insects'
                    metric_label = "total insect count"
                    metric_unit = "insects"
                
                # Find result based on comparison type
                comparison = intent.get("comparison_type", "").lower()
                if comparison in ["lowest", "least", "minimum", "smallest", "fewest"]:
                    result = min(all_stats, key=lambda x: x[metric_key])
                    comparison_word = "lowest"
                    sorted_stats = sorted(all_stats, key=lambda x: x[metric_key])
                elif comparison in ["highest", "most", "maximum", "largest", "greatest"]:
                    result = max(all_stats, key=lambda x: x[metric_key])
                    comparison_word = "highest"
                    sorted_stats = sorted(all_stats, key=lambda x: x[metric_key], reverse=True)
                else:
                    # Compare all
                    sorted_stats = sorted(all_stats, key=lambda x: x[metric_key], reverse=True)
                    result = sorted_stats[0]
                    comparison_word = "highest"
                
                # Format answer
                if metric_key == 'density':
                    answer = f"**{result['location']}** has the **{comparison_word} {metric_label}** with **{result['density']} {metric_unit}** ({result['total_insects']} total insect(s) across {result['image_count']} images)."
                else:
                    answer = f"**{result['location']}** has the **{comparison_word} {metric_label}** with **{result['total_insects']} {metric_unit}** across {result['image_count']} image(s)."
                
                # Build verification data
                all_data = "\n".join([
                    f"{i+1}. **{s['location']}**: {s['total_insects']} insects ({s['density']} per 100 images, {s['image_count']} reports)"
                    for i, s in enumerate(sorted_stats)
                ])
                
                return {
                    "answer": answer,
                    "all_data": all_data,
                    "show_all_data": True,
                    "validated": True,
                    "source": "reports_table",
                    "semantic_intent": intent
                }
            
            # Handle specific location queries
            elif intent.get("query_type") == "specific_location" and intent.get("location"):
                location = intent.get("location")
                cursor.execute("""
                    SELECT COUNT(*) as image_count, SUM(insect_count) as total_insects
                    FROM reports 
                    WHERE location = ?
                """, (location,))
                row = cursor.fetchone()
                
                if row and row[0] > 0:
                    img_count, total_insects = row[0], row[1] or 0
                    density = (total_insects / img_count * 100) if img_count > 0 else 0
                    
                    answer = f"**{location}** has **{total_insects} insects** across **{img_count} images**."
                    
                    return {
                        "answer": answer,
                        "show_all_data": False,
                        "validated": True,
                        "source": "reports_table",
                        "semantic_intent": intent
                    }
            
            # Handle count queries (total across all locations)
            elif intent.get("query_type") == "count":
                # Check if asking for count of locations/cities
                if intent.get("metric") == "locations":
                    cursor.execute("SELECT COUNT(DISTINCT location) as location_count FROM reports")
                    row = cursor.fetchone()
                    if row:
                        location_count = row[0]
                        answer = f"You have visited **{location_count} cities/locations** for collecting stink bugs."
                        
                        # Show all locations
                        cursor.execute("SELECT DISTINCT location FROM reports ORDER BY location")
                        locations = [loc[0] for loc in cursor.fetchall()]
                        all_data = "\n".join([f"{i+1}. {loc}" for i, loc in enumerate(locations)])
                        
                        return {
                            "answer": answer,
                            "all_data": all_data,
                            "show_all_data": True,
                            "validated": True,
                            "source": "reports_table",
                            "semantic_intent": intent
                        }
                else:
                    # General count query
                    cursor.execute("""
                        SELECT COUNT(DISTINCT location) as locations, 
                               COUNT(*) as total_images, 
                               SUM(insect_count) as total_insects
                        FROM reports
                    """)
                    row = cursor.fetchone()
                    
                    if row:
                        locations, total_images, total_insects = row[0], row[1], row[2] or 0
                        avg_density = (total_insects / total_images * 100) if total_images > 0 else 0
                        
                        answer = f"Across **{locations} locations**, you have **{total_insects} total insects** in **{total_images} images**."
                        
                        return {
                            "answer": answer,
                            "show_all_data": False,
                            "validated": True,
                            "source": "reports_table",
                            "semantic_intent": intent
                        }
                    
    except Exception as e:
        print(f"Database query error: {e}")
        return None
    
    return None


def get_learned_answer(user_query: str, threshold: float = 0.6) -> tuple[str, str] | None:
    """
    Check if there's a similar question with thumbs-up feedback in the database.
    Returns (cached_answer, response_type) if match found, None otherwise.
    Uses simple word overlap similarity for matching.
    IMPORTANT: Detects contradictory queries (lowest vs highest) and avoids false matches.
    """
    if not user_query or len(user_query.strip()) < 3:
        return None
    
    # Normalize query
    query_lower = user_query.lower()
    query_words = set(query_lower.split())
    if len(query_words) == 0:
        return None
    
    # Detect query intent to avoid contradictions
    is_lowest_query = any(word in query_lower for word in ["lowest", "least", "fewest", "minimum", "smallest"])
    is_highest_query = any(word in query_lower for word in ["highest", "most", "greatest", "maximum", "largest"])
    
    # Get all thumbs-up feedback
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_query, assistant_response, response_type FROM feedback WHERE label = 'yes' ORDER BY created_at DESC LIMIT 100"
        )
        rows = cursor.fetchall()
    
    best_match = None
    best_score = 0.0
    
    for stored_query, stored_answer, resp_type in rows:
        if not stored_query:
            continue
        
        stored_lower = stored_query.lower()
        stored_words = set(stored_lower.split())
        if len(stored_words) == 0:
            continue
        
        # Check for contradictory intent (asking opposite question)
        stored_is_lowest = any(word in stored_lower for word in ["lowest", "least", "fewest", "minimum", "smallest"])
        stored_is_highest = any(word in stored_lower for word in ["highest", "most", "greatest", "maximum", "largest"])
        
        # Skip if intents are opposite
        if (is_lowest_query and stored_is_highest) or (is_highest_query and stored_is_lowest):
            continue  # Don't match opposite queries!
        
        # Also check if the answer content contradicts the query
        answer_lower = stored_answer.lower()
        if is_lowest_query and any(word in answer_lower for word in ["highest", "most", "maximum", "largest", "greatest"]):
            continue  # Answer talks about highest but query asks for lowest
        if is_highest_query and any(word in answer_lower for word in ["lowest", "least", "minimum", "smallest", "fewest"]):
            continue  # Answer talks about lowest but query asks for highest
        
        # Simple Jaccard similarity (intersection over union)
        intersection = len(query_words & stored_words)
        union = len(query_words | stored_words)
        similarity = intersection / union if union > 0 else 0.0
        
        if similarity > best_score:
            best_score = similarity
            best_match = (stored_query, stored_answer, resp_type)
    
    if best_score >= threshold and best_match:
        return (best_match[1], best_match[2])  # Return answer and type
    
    return None


def show_feedback_buttons(msg_idx: int):
    """Render feedback buttons immediately after an assistant message."""
    # Check if this message already has feedback
    existing_feedback = next((fb for fb in st.session_state.get("feedback", []) if fb.get("msg_idx") == msg_idx), None)
    
    if existing_feedback:
        # Show a simple thank you message with the rating
        feedback_emoji = "👍" if existing_feedback.get("label") == "yes" else "👎"
        st.caption(f"Thanks for your feedback! {feedback_emoji}")
    else:
        # Show feedback buttons
        col1, col2, col3 = st.columns([0.35, 0.08, 0.08])
        with col1:
            st.caption("👉 Was this helpful?")
        with col2:
            if st.button("👍", key=f"yes_{msg_idx}", help="Mark as helpful"):
                st.session_state["feedback"].append({"msg_idx": msg_idx, "label": "yes"})
                
                if msg_idx < len(st.session_state["messages"]):
                    assistant_msg = st.session_state["messages"][msg_idx]
                    user_msg = st.session_state["messages"][msg_idx - 1] if msg_idx > 0 else {"content": ""}
                    
                    save_feedback_to_db(
                        user_query=user_msg.get("content", ""),
                        assistant_response=assistant_msg.get("content", ""),
                        label="yes",
                        response_type=assistant_msg.get("response_type", "unknown")
                    )
                st.rerun()
        with col3:
            if st.button("👎", key=f"no_{msg_idx}", help="Mark as not helpful"):
                st.session_state["feedback"].append({"msg_idx": msg_idx, "label": "no"})
                
                if msg_idx < len(st.session_state["messages"]):
                    assistant_msg = st.session_state["messages"][msg_idx]
                    user_msg = st.session_state["messages"][msg_idx - 1] if msg_idx > 0 else {"content": ""}
                    
                    save_feedback_to_db(
                        user_query=user_msg.get("content", ""),
                        assistant_response=assistant_msg.get("content", ""),
                        label="no",
                        response_type=assistant_msg.get("response_type", "unknown")
                    )
                st.rerun()


def render_chat_interface(location="main"):
    """
    Renders the Streamlit chat interface, including chat history display and user input.
    Supports both text queries and image uploads with YOLO detection.
    Returns the user's input if submitted, otherwise None.
    """
    st.subheader("Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "feedback" not in st.session_state:
        st.session_state["feedback"] = []  # list of {msg_index, role, type, content, label, ts}

    # Display chat messages from history on app rerun
    for idx, message in enumerate(st.session_state["messages"]):
        avatar_icon = "🐞" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar_icon):
            if message["type"] == "text":
                st.write(message["content"])
            elif message["type"] == "image":
                # Replay image messages using stored fast_image data
                if "fast_image" in message:
                    fast_image = message["fast_image"]
                    if fast_image.get('is_plural'):  # Multiple images
                        st.markdown(f"Here are the images from {fast_image['location']}:")
                        cols = st.columns(3)
                        for img_idx, img in enumerate(fast_image['images']):
                            with cols[img_idx % 3]:
                                st.image(img['image_bytes'], caption=img['caption'], use_container_width=True)
                    else:  # Single image
                        st.markdown(f"Here is the image from {fast_image['location']}:")
                        st.image(fast_image['image_bytes'], caption=fast_image['caption'], use_container_width=True)
                elif "image_data" in message:
                    # Legacy format support
                    st.image(message["image_data"], caption=message.get("caption", ""), use_container_width=True)
        
        # Show feedback buttons for assistant messages
        if message["role"] == "assistant":
            show_feedback_buttons(idx)

    # Accept user input with unique key based on location
    col_input, col_upload_btn = st.columns([0.95, 0.05])
    
    with col_input:
        user_input = st.chat_input("Ask me anything about your RBSB reports...", key=f"chat_input_{location}")
    
    with col_upload_btn:
        show_upload = st.button("⬆️", key=f"upload_toggle_input_{location}", help="Upload image", use_container_width=True)
        if show_upload:
            st.session_state[f"show_upload_{location}"] = not st.session_state.get(f"show_upload_{location}", False)
            st.rerun()

    # # Reference Images Section
    # with st.expander("📚 Stinkbug Reference Images & Resources", expanded=False):
    #     st.markdown("**Educational Resources:**")
    #     col1, col2, col3 = st.columns(3)
        
    #     with col1:
    #         st.markdown("[**BugGuide**](https://www.bugguide.net/node/view/15740)\nRedbanded Stinkbug (RBSB) identification guide")
        
    #     with col2:
    #         st.markdown("[**iNaturalist**](https://www.inaturalist.org/taxa/search?q=stinkbug)\nStinkbug observations worldwide")
        
    #     with col3:
    #         st.markdown("[**LSU AgCenter**](https://www.lsuagcenter.com/)\nLouisiana stinkbug management info")
        
    #     st.divider()
    #     st.markdown("**Quick Reference - RBSB Characteristics:**")
    #     st.markdown("""
    #     - **Color**: Reddish-brown to dark brown
    #     - **Size**: 12-17mm (about the size of a dime)
    #     - **Shape**: Shield-shaped (pentagonal)
    #     - **Antennae**: Brown with light bands
    #     - **Scent**: Distinctive unpleasant odor when crushed
    #     - **Habitat**: Field crops, vegetables, fruits
    #     """)

    if st.session_state.get(f"show_upload_{location}", False):
        st.divider()
        st.markdown("**Upload & Analyze**")
        # SAM 2 Verification toggle in chat
        verifier = get_sam2_verifier()
        sam2_status = verifier.get_status()
        
        col1, col2, col3 = st.columns([0.33, 0.33, 0.34])
        with col1:
            if sam2_status['available']:
                use_sam2_chat = st.checkbox(
                    "SAM 2 Verification",
                    value=False,
                    key=f"sam2_chat_{location}",
                    help="Double-check counts"
                )
            else:
                st.info("SAM 2 unavailable")
                use_sam2_chat = False

        with col2:
            ground_status_chat = get_grounding_status()
            if ground_status_chat['available']:
                text_filter_chat = st.checkbox(
                    "Text-prompt filter",
                    value=False,
                    key=f"text_filter_chat_{location}",
                    help="Filter with GroundingDINO"
                )
            else:
                st.caption("GroundingDINO N/A")
                text_filter_chat = False

        with col3:
            if text_filter_chat:
                text_prompt_chat = st.text_input("Prompt", value="stinkbug", max_chars=60, key=f"text_prompt_chat_{location}")
            else:
                text_prompt_chat = ""

        uploaded_image = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"],
            key=f"chat_image_upload_{location}"
        )

        if uploaded_image:
            with st.spinner("Analyzing..."):
                image_bytes = uploaded_image.getvalue()
                insect_count, annotated_bytes, metadata = count_insects(
                    image_bytes,
                    use_sam2_verification=use_sam2_chat,
                    text_filter_enabled=text_filter_chat,
                    text_prompt=text_prompt_chat,
                    iou_threshold=0.5
                )

                # Store detection in session for chat
                detection_result = {
                    "image_name": uploaded_image.name,
                    "count": insect_count,
                    "annotated": annotated_bytes,
                    "metadata": metadata,
                }

                # Display results
                st.success(f"✓ **{insect_count} insects** detected")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(annotated_bytes, caption=f"Detected: {insect_count}", use_container_width=True)
                with col2:
                    st.image(image_bytes, caption="Original", use_container_width=True)

                # Show SAM2 verification details if used
                if metadata.get('verified'):
                    with st.expander("SAM 2 Details"):
                        st.write(f"YOLO: {metadata.get('yolo_count', 0)} | Verified: {metadata.get('sam2_count', 0)}")
                        st.write(f"Quality: {metadata.get('annotation_quality', 'Unknown')}")
                        if metadata.get('quality_issues'):
                            st.warning(f"{len(metadata.get('quality_issues', []))} issues found")
                        else:
                            st.success("Quality check passed")

                # Option to save to reports
                with st.form(f"save_detection_form_{location}"):
                    save_location = st.text_input(
                        "Location",
                        max_chars=120,
                        key=f"chat_save_location_{location}"
                    )
                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        save_button = st.form_submit_button("Save to Reports", use_container_width=True)
                    with col_cancel:
                        cancel_button = st.form_submit_button("Discard", use_container_width=True)

                if save_button and save_location.strip():
                    user = st.session_state["auth_user"]
                    insert_reports(user, save_location.strip(), [detection_result])
                    threshold_check(save_location.strip(), user["email"])
                    update_gallery(save_location.strip(), [detection_result])
                    
                    # Add to chat as assistant message
                    ai_msg = f"Saved! Location: {save_location.strip()}, Insects: {insect_count}"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ai_msg,
                        "type": "text"
                    })
                    st.success("Report saved!")
                    st.session_state[f"show_upload_{location}"] = False

    return user_input


def main():
    st.set_page_config(page_title="Redbanded Stink Bug Analytics", page_icon="🐞", layout="wide")
    ensure_session_defaults()
    init_db()
    inject_global_styles()
    configure_plotly_template()

    if not st.session_state["auth_user"]:
        render_login_register()
        return



    with st.sidebar:
        st.title("RBSB Analytics")
        menu_options = ["Homepage", "Capture Image", "Reports", "Search Images", "Chat", "Feedback Analytics"]
        st.session_state["menu_choice"] = st.radio(
            "",
            key="nav_menu",
            options=menu_options,
            index=menu_options.index(st.session_state["menu_choice"]) if st.session_state["menu_choice"] in menu_options else 0
        )
        
        # Agentic Mode Settings (only show in chat)
        if st.session_state["menu_choice"] == "Chat":
            st.divider()
            st.subheader("🤖 AI Settings")
            agent_mode = st.checkbox(
                "Advanced Agent Mode",
                value=st.session_state.get("agent_mode", False),
                help="Enable autonomous agent with tool use, planning, and self-correction"
            )
            st.session_state["agent_mode"] = agent_mode
            
            if agent_mode:
                show_reasoning = st.checkbox(
                    "Show Reasoning Steps",
                    value=st.session_state.get("show_agent_reasoning", False),
                    help="Display the agent's chain-of-thought process"
                )
                st.session_state["show_agent_reasoning"] = show_reasoning
                
                st.caption("Agent will:")
                st.caption("• Create multi-step plans")
                st.caption("• Query database autonomously")
                st.caption("• Self-correct errors")
                st.caption("• Use tools in parallel")
        
        st.divider()
        st.button("Logout", on_click=logout)


    choice = st.session_state["menu_choice"]
    if choice == "Homepage":
        render_home()
    elif choice == "Capture Image":
        render_capture()
    elif choice == "Reports":
        render_reports()
    elif choice == "Search Images":
        render_search()
    elif choice == "Feedback Analytics":
        render_feedback_analytics()
    elif choice == "Chat":
        user_input = render_chat_interface(location="main")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input, "type": "text"})
            with st.chat_message("user", avatar="🐞"):
                st.markdown(user_input)

            # ===== MASTER CHECK: Does this question require reasoning/analysis? =====
            # If yes, skip ALL fast paths and go straight to Agent/LLM
            query_lower = user_input.lower()
            requires_reasoning = any(word in query_lower for word in [
                "why", "reason", "explain", "understand", "effect", "impact", "consequence",
                "cause", "because", "analysis", "analyze", "interpret", "mean", "suggest",
                "recommend", "advise", "solution", "help me", "what could be"
            ])
            
            print(f"\n=== MASTER CHECK ===")
            print(f"User input: {user_input[:100]}...")
            print(f"requires_reasoning: {requires_reasoning}")
            
            if requires_reasoning:
                # Skip all fast paths - go straight to reasoning with data
                db_answer = None
                print(f"SKIPPING database path - requires reasoning")
            else:
                # ===== PRIORITY 1: DATABASE-FIRST QUERIES (VALIDATED DATA) =====
                # Check if this is a database-answerable query (comparisons, stats, etc.)
                db_answer = try_database_answer(user_input)
                print(f"CHECKING database path - db_answer is {'NOT None' if db_answer else 'None'}")
            
            if db_answer:
                with st.chat_message("assistant", avatar="🤖"):
                    st.success("**Validated Answer** - Direct from your reports database")
                    st.markdown(db_answer["answer"])
                    if db_answer.get("show_all_data"):
                        with st.expander("See All Location Data"):
                            st.markdown(db_answer["all_data"])
                msg_idx = len(st.session_state.messages)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": db_answer["answer"],
                    "type": "text",
                    "response_type": "database_validated"
                })
                show_feedback_buttons(msg_idx)
                return

            # ===== PRIORITY 2: FAST DETERMINISTIC ANSWERS =====
            # Fast deterministic answer (no model call)
            fast_response = fast_answer(user_input)
            print(f"Priority 2 fast_answer(): {'returned response' if fast_response else 'returned None'}")
            if fast_response:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(fast_response)
                msg_idx = len(st.session_state.messages)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": fast_response,
                    "type": "text",
                    "response_type": "fast_answer"
                })
                show_feedback_buttons(msg_idx)
                return

            # ===== PRIORITY 3: FAST IMAGE RESPONSES =====
            # Fast deterministic image fetch by location (no model call)
            fast_image = fast_image_response(user_input)
            print(f"Priority 3 fast_image_response(): {'returned images' if fast_image else 'returned None'}")
            if fast_image:
                with st.chat_message("assistant", avatar="🤖"):
                    if fast_image.get('filter_note'):
                        st.warning(fast_image['filter_note'])
                    if fast_image.get('is_plural'):  # Multiple images
                        st.markdown(f"Here are the images from {fast_image['location']}:")
                        cols = st.columns(3)
                        for idx, img in enumerate(fast_image['images']):
                            with cols[idx % 3]:
                                st.image(img['image_bytes'], caption=img['caption'], use_container_width=True)
                    else:  # Single image
                        st.markdown(f"Here is the image from {fast_image['location']}:")
                        st.image(fast_image['image_bytes'], caption=fast_image['caption'], use_container_width=True)
                msg_idx = len(st.session_state.messages)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Images from {fast_image['location']}",
                    "type": "image",
                    "fast_image": fast_image,  # Store the full image data for replay
                    "response_type": "fast_image"
                })
                show_feedback_buttons(msg_idx)
                return
            
            # ============ ADVANCED AGENT MODE ============
            if st.session_state.get("agent_mode", False) and st.session_state["ollama_client"]:
                print(f"Priority 4: ENTERING AGENT MODE")
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("🤖 Agent analyzing and planning..."):
                        try:
                            # Try mistral first, fallback to llama2
                            agent_model = 'mistral'
                            try:
                                # Test if mistral is available
                                st.session_state["ollama_client"].chat(
                                    model='mistral',
                                    messages=[{'role': 'user', 'content': 'test'}],
                                    options={"num_predict": 1}
                                )
                            except:
                                # Mistral not available, use llama2
                                agent_model = 'llama2'
                                print(f"  Mistral not found, using llama2 instead")
                            
                            # Initialize agent
                            agent = AdvancedAgent(
                                ollama_client=st.session_state["ollama_client"],
                                model=agent_model,
                                verbose=st.session_state.get("show_agent_reasoning", False)
                            )
                            
                            # Run agent
                            result = agent.run(
                                user_query=user_input,
                                context=get_response_context()
                            )
                            
                            # Show validation badge if using direct database data
                            if result.get("validated"):
                                st.success("✅ **Validated Answer** - Direct database query (no AI interpretation)")
                            
                            # Display reasoning steps if enabled
                            if st.session_state.get("show_agent_reasoning", False) and result.get("execution_history"):
                                with st.expander("🧠 Agent Reasoning Process", expanded=True):
                                    for step in result["execution_history"]:
                                        st.markdown(f"**Step {step['step']}**: {step['thought'][:150]}...")
                                        st.caption(f"Action: {step['action']}")
                                        if step['tool']:
                                            status_icon = "✅" if step['status'] == "success" else "❌"
                                            st.caption(f"{status_icon} Tool: `{step['tool']}`")
                                    
                                    st.divider()
                                    st.caption(f"🔧 Tools used: {', '.join(result.get('tools_used', []))}")
                                    st.caption(f"📊 Steps executed: {result.get('steps_executed', 0)}")
                            
                            # Display answer
                            st.markdown(result.get("answer", "I couldn't generate an answer."))
                            
                            # Store in session
                            msg_idx = len(st.session_state.messages)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": result.get("answer", ""),
                                "type": "text",
                                "response_type": "agent",
                                "agent_metadata": {
                                    "tools_used": result.get("tools_used", []),
                                    "steps": result.get("steps_executed", 0)
                                }
                            })
                            
                            show_feedback_buttons(msg_idx)
                            return
                            
                        except Exception as e:
                            # st.error(f"Agent error: {e}")
                            st.info("Don't forget to give feedback on the response below!")
            
            # ============ END AGENT MODE ============

            print(f"Priority 5: FALLING BACK TO LLM (agent mode not enabled or client not available)")
            with st.chat_message("assistant", avatar="🤖"):
                if st.session_state["ollama_client"]:
                    with st.spinner("Thinking..."):
                        prompt = generate_ollama_prompt(user_input)
                        print(f"\n--- PROMPT TO OLLAMA ---\n{prompt}\n--- END PROMPT ---\n") # Debug print
                        try:
                            # Try with Mistral first (better instruction following), fallback to llama2
                            model_to_use = 'mistral'
                            try:
                                response = st.session_state["ollama_client"].chat(
                                    model=model_to_use,
                                    messages=[{'role': 'user', 'content': prompt}],
                                    options={"temperature": 0.1, "num_predict": 512, "num_ctx": 4096}
                                )
                            except:
                                # Fallback to llama2 if mistral not available
                                model_to_use = 'llama2'
                                response = st.session_state["ollama_client"].chat(
                                    model=model_to_use,
                                    messages=[{'role': 'user', 'content': prompt}],
                                    options={"temperature": 0.1, "num_predict": 512, "num_ctx": 4096}
                                )
                            
                            ai_response = response['message']['content']
                            print(f"\n--- RAW OLLAMA RESPONSE (Model: {model_to_use}) ---\n{ai_response}\n--- END RAW OLLAMA RESPONSE ---\n") # Debug print

                            # Validate response grounding
                            is_grounded, warning = validate_response_grounding(ai_response, user_input, get_response_context())
                            if not is_grounded:
                                st.warning(warning)

                            # Retrieve all gallery items once for efficient lookup
                            all_gallery_items = {item['image_name']: {'image_bytes': item['image_bytes'], 'location': item['location'], 'timestamp': item['timestamp']} for item in get_gallery_data()}

                            # Define the regex pattern for the image tag and optional trailing text (caption)
                            # Group 1: full tag string e.g., [DISPLAY_IMAGE:image.jpg]
                            # Group 2: image_name e.g., image.jpg
                            # Group 3: potential_caption_text (any text after the tag, before next tag or end of string)
                            image_tag_with_caption_pattern = r"(\[DISPLAY_IMAGE:([^\]]+?)\])\s*(.*?)(?=\[DISPLAY_IMAGE:|$)"

                            processed_ollama_output = []
                            last_match_end = 0

                            # Find all matches of the image tag with potential trailing caption
                            for match in re.finditer(image_tag_with_caption_pattern, ai_response):
                                # Add any text that appeared before this match as a separate text message
                                text_before_match = ai_response[last_match_end : match.start()].strip()
                                if text_before_match:
                                    processed_ollama_output.append({"type": "text", "content": text_before_match})

                                full_tag_string = match.group(1) # e.g., [DISPLAY_IMAGE:image.jpg]
                                image_name = match.group(2).strip() # e.g., image.jpg
                                potential_caption_text = match.group(3).strip() # any text after the tag, before next tag or end

                                image_info = all_gallery_items.get(image_name)

                                if image_info and image_info['image_bytes']: # Ensure image exists and has bytes
                                    # Create a comprehensive caption for display using image info and AI's text
                                    caption_for_display = f"Image: {image_name}"
                                    if image_info.get('location'):
                                        caption_for_display += f" from {image_info['location']}"
                                    if image_info.get('timestamp'):
                                        caption_for_display += f" captured at {image_info['timestamp']}"
                                    if potential_caption_text:
                                        caption_for_display += f" - {potential_caption_text}"

                                    processed_ollama_output.append({
                                        "type": "image",
                                        "image_data": image_info['image_bytes'],
                                        "caption": caption_for_display,
                                        "content": "" # All textual content for the image is in the 'caption'
                                    })
                                else:
                                    # If image not found or no bytes, treat the whole matched string as text
                                    not_found_message = f" (Image '{image_name}' not found or invalid.)" if not image_info else ""
                                    text_content = f"{full_tag_string} {potential_caption_text}{not_found_message}".strip()
                                    processed_ollama_output.append({"type": "text", "content": text_content})

                                last_match_end = match.end()

                            # Add any remaining text after the last image tag
                            remaining_text = ai_response[last_match_end:].strip()
                            if remaining_text:
                                processed_ollama_output.append({"type": "text", "content": remaining_text})

                            # Display parts and store in session state
                            for part in processed_ollama_output:
                                if part["type"] == "text" and part["content"]:
                                    st.markdown(part["content"])
                                    st.session_state.messages.append({"role": "assistant", "content": part["content"], "type": "text", "response_type": "ollama"})
                                elif part["type"] == "image":
                                    st.image(part["image_data"], caption=part["caption"], use_container_width=True)
                                    # For image messages, the 'content' field in session state is empty,
                                    # as the full descriptive text is handled by 'caption' of st.image.
                                    st.session_state.messages.append({"role": "assistant", "content": part["content"], "type": "image", "image_data": part["image_data"], "caption": part["caption"], "response_type": "ollama"})

                            # Show feedback for the latest assistant message from this response
                            if st.session_state.messages:
                                show_feedback_buttons(len(st.session_state.messages) - 1)

                        except ollama.ResponseError as e:
                            ai_response_error = f"Error: {e}. Failed to get response from Ollama. Make sure 'llama2' model is pulled."
                            st.error(ai_response_error)
                            msg_idx = len(st.session_state.messages)
                            st.session_state.messages.append({"role": "assistant", "content": ai_response_error, "type": "text", "response_type": "error"})
                            show_feedback_buttons(msg_idx)
                            return
                        except Exception as e:
                            ai_response_error = f"An unexpected error occurred: {e}"
                            st.error(ai_response_error)
                            msg_idx = len(st.session_state.messages)
                            st.session_state.messages.append({"role": "assistant", "content": ai_response_error, "type": "text", "response_type": "error"})
                            show_feedback_buttons(msg_idx)
                            return
                else:
                    ai_response_not_init = "Ollama client not initialized. Cannot answer questions."
                    st.error(ai_response_not_init)
                    msg_idx = len(st.session_state.messages)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response_not_init, "type": "text", "response_type": "error"})
                    show_feedback_buttons(msg_idx)
                    return


if __name__ == "__main__":
    main()