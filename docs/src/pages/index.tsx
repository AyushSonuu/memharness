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

# 1. Create memory harness (SQLite for dev, PostgreSQL for prod)
harness = MemoryHarness("sqlite:///agent_memory.db")
await harness.connect()

# 2. Get read-only memory tools for agent self-awareness
memory_tools = get_read_tools(harness)

# 3. Create a memory-aware agent
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",  # any LLM via init_chat_model
    tools=memory_tools + your_other_tools,
    system_prompt="""You are a helpful assistant with persistent memory.
Use your memory tools to search and read important information.
Before answering, search your memory for relevant context.""",
)

# 4. Agent can now search, read, and assemble context...
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "What did we discuss yesterday?"}]
})`;

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
