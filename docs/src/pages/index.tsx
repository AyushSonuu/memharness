import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import CodeBlock from '@theme/CodeBlock';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          🧠 {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/getting-started">
            Get Started →
          </Link>
          <Link
            className="button button--outline button--lg"
            style={{marginLeft: '1rem', color: 'white', borderColor: 'white'}}
            href="https://pypi.org/project/memharness/">
            pip install memharness
          </Link>
        </div>
      </div>
    </header>
  );
}

type FeatureItem = {
  title: string;
  emoji: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: '8 Memory Types',
    emoji: '📦',
    description: (
      <>
        Conversational, Knowledge Base, Entity, Workflow, Toolbox, Summary,
        Tool Log, and Persona — each with its own schema, storage strategy,
        and retrieval pattern.
      </>
    ),
  },
  {
    title: 'Framework Agnostic',
    emoji: '🔌',
    description: (
      <>
        Works with LangChain, LangGraph, CrewAI, Deep Agents, or your custom
        agent framework. One memory layer, any agent.
      </>
    ),
  },
  {
    title: 'Pluggable Backends',
    emoji: '🗄️',
    description: (
      <>
        PostgreSQL + pgvector for production, SQLite for development, in-memory
        for testing. Same API across all backends.
      </>
    ),
  },
  {
    title: '5 Read-Only Tools',
    emoji: '🔍',
    description: (
      <>
        Search memory, read by ID, expand summaries, assemble context,
        and discover tools — all as LangChain BaseTool subclasses.
        Middleware handles writes.
      </>
    ),
  },
  {
    title: 'Memory Lifecycle',
    emoji: '♻️',
    description: (
      <>
        Built-in agents for summarization, consolidation, entity extraction,
        and garbage collection. Configurable policies keep memory clean.
      </>
    ),
  },
  {
    title: 'Async-First API',
    emoji: '⚡',
    description: (
      <>
        Full async/await support. Context managers, type hints, Pydantic
        config models. Python 3.13+.
      </>
    ),
  },
];

function Feature({title, emoji, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center" style={{fontSize: '3rem', marginBottom: '0.5rem'}}>
        {emoji}
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

const agentExample = `from memharness import MemoryHarness
from memharness.tools import get_read_tools
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from memharness.agents import ContextAssemblyAgent

# --- BEFORE middleware: inject context + load conversation ---
class ContextMiddleware(AgentMiddleware):
    def __init__(self, harness, thread_id):
        super().__init__()
        self.harness = harness
        self.tid = thread_id
        self._ctx = ContextAssemblyAgent(harness)
        self._loaded = 0

    async def abefore_model(self, state, runtime):
        msgs = state.get("messages", [])
        query = next((m.content for m in reversed(msgs) if isinstance(m, HumanMessage)), "")
        if not query: return None
        # Save user message
        await self.harness.add_conversational(self.tid, "user", query)
        # Assemble context (KB, entities, workflows, persona)
        ctx = await self._ctx.assemble(query=query, thread_id=self.tid)
        return {"messages": ctx.to_messages()}

    async def aafter_model(self, state, runtime):
        # Save assistant response
        msgs = state.get("messages", [])
        last = msgs[-1] if msgs else None
        if isinstance(last, AIMessage) and last.content:
            await self.harness.add_conversational(self.tid, "assistant", last.content)
        return None

# --- Setup ---
harness = MemoryHarness("sqlite:///agent_memory.db")
await harness.connect()

thread_id = "user-alice"
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=get_read_tools(harness),       # 5 read-only tools
    middleware=[
        ContextMiddleware(harness, thread_id),          # BEFORE: context + save msgs
        # Add your own AFTER middleware for entities + workflow (see docs)
    ],
)

# Turn 1: agent saves this to conv table via middleware
r1 = await agent.ainvoke({"messages": [{"role": "user", "content": "I work at SAP"}]})

# Turn 2: middleware loads past messages — agent remembers!
r2 = await agent.ainvoke({"messages": [{"role": "user", "content": "Where do I work?"}]})
# → "You work at SAP"  (loaded from conversation memory)`;

const standaloneExample = `from memharness import MemoryHarness
from memharness.agents import ContextAssemblyAgent

async with MemoryHarness("sqlite:///memory.db") as harness:
    # Store memories across types
    await harness.add_conversational("thread-1", "user", "I prefer Python")
    await harness.add_knowledge("Python 3.13 has free-threading", source="docs")
    await harness.add_entity("Alice", "PERSON", "Engineer at Acme Corp")
    await harness.add_workflow(
        task="Deploy app",
        steps=["Build", "Test", "Docker push", "K8s apply"],
        outcome="Deployed successfully",
    )

    # Assemble context (BEFORE-loop pattern from agent memory course)
    ctx_agent = ContextAssemblyAgent(harness)
    ctx = await ctx_agent.assemble("Tell me about Python", thread_id="thread-1")

    # Get as LangChain messages (SystemMessage + HumanMessage + AIMessage)
    messages = ctx.to_messages()  # list[BaseMessage]

    # Or as markdown prompt string
    prompt = ctx.to_prompt()  # str with ## sections`;

function QuickStart() {
  return (
    <section style={{padding: '2rem 0', backgroundColor: 'var(--ifm-background-surface-color)'}}>
      <div className="container">
        <div className="row">
          <div className="col col--10 col--offset-1">
            <Heading as="h2" className="text--center" style={{marginBottom: '0.5rem'}}>
              🤖 Use with LangChain Agent
            </Heading>
            <p className="text--center" style={{color: 'var(--ifm-color-secondary-darkest)', marginBottom: '1rem'}}>
              Give any agent persistent, searchable memory with 5 read-only tools
            </p>
            <CodeBlock language="python" title="langchain_agent.py" showLineNumbers>
              {agentExample}
            </CodeBlock>
          </div>
        </div>
        <div className="row" style={{marginTop: '2rem'}}>
          <div className="col col--10 col--offset-1">
            <Heading as="h2" className="text--center" style={{marginBottom: '0.5rem'}}>
              🧠 Or Use Standalone
            </Heading>
            <p className="text--center" style={{color: 'var(--ifm-color-secondary-darkest)', marginBottom: '1rem'}}>
              No framework needed — memharness works with any Python code
            </p>
            <CodeBlock language="python" title="standalone.py" showLineNumbers>
              {standaloneExample}
            </CodeBlock>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  return (
    <Layout
      title="Memory Infrastructure for AI Agents"
      description="Framework-agnostic memory infrastructure for AI agents. 8 memory types, pluggable backends, 5 read-only tools.">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              {FeatureList.map((props, idx) => (
                <Feature key={idx} {...props} />
              ))}
            </div>
          </div>
        </section>
        <QuickStart />
      </main>
    </Layout>
  );
}
