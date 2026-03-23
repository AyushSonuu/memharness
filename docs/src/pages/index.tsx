import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

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
    title: '10 Memory Types',
    emoji: '📦',
    description: (
      <>
        Conversational, Knowledge Base, Entity, Workflow, Toolbox, Summary,
        Tool Log, Skills, File, and Persona — each with its own schema,
        storage strategy, and retrieval pattern.
      </>
    ),
  },
  {
    title: 'Framework Agnostic',
    emoji: '🔌',
    description: (
      <>
        Works with LangChain, LangGraph, CrewAI, Deep Agents, or your custom
        agent framework. One memory layer, any agent. Includes built-in
        LangChain and LangGraph adapters.
      </>
    ),
  },
  {
    title: 'Pluggable Backends',
    emoji: '🗄️',
    description: (
      <>
        PostgreSQL + pgvector for production, SQLite for development, in-memory
        for testing. Same API across all backends. Swap with one line of config.
      </>
    ),
  },
  {
    title: 'Async-First API',
    emoji: '⚡',
    description: (
      <>
        Built for modern Python. Full async/await support throughout. Context
        managers, type hints, and Pydantic models for configuration.
      </>
    ),
  },
  {
    title: 'Memory Lifecycle',
    emoji: '♻️',
    description: (
      <>
        Built-in agents for summarization, consolidation, garbage collection,
        and entity extraction. Configurable policies keep your memory clean
        and relevant.
      </>
    ),
  },
  {
    title: 'Self-Exploration Tools',
    emoji: '🔍',
    description: (
      <>
        Agents can explore their own memory with built-in LangChain tools.
        VFS-based tool discovery with tree, ls, grep, and cat operations
        for the Toolbox memory type.
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

function QuickStart() {
  return (
    <section style={{padding: '2rem 0', backgroundColor: 'var(--ifm-background-surface-color)'}}>
      <div className="container">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <Heading as="h2" className="text--center" style={{marginBottom: '0.5rem'}}>
              Use with LangChain Agent
            </Heading>
            <p className="text--center" style={{color: 'var(--ifm-color-secondary-darkest)', marginBottom: '1.5rem'}}>
              Give any agent persistent, searchable memory in 10 lines
            </p>
            <pre style={{
              padding: '1.5rem',
              borderRadius: '8px',
              fontSize: '0.85rem',
              lineHeight: '1.6',
            }}>
              <code>{`from memharness import MemoryHarness
from memharness.tools import get_memory_tools
from langchain.agents import create_agent

# 1. Create memory harness
harness = MemoryHarness("sqlite:///agent_memory.db")
await harness.connect()

# 2. Get 12 memory tools (search, read, write, expand, summarize, etc.)
memory_tools = get_memory_tools(harness)

# 3. Create a memory-aware agent
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",  # or any LLM
    tools=memory_tools + your_other_tools,
    system_prompt="""You are a helpful assistant with persistent memory.
Use your memory tools to remember important information across conversations.
Before answering, search your memory for relevant context.""",
)

# 4. The agent can now:
#    - Search past conversations, knowledge, entities, workflows
#    - Write important facts to knowledge base
#    - Log tool executions for audit trail
#    - Save successful task patterns as reusable workflows
#    - Expand compacted summaries for full detail
#    - Assemble complete context from all memory types
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "What did we discuss yesterday?"}]
})`}</code>
            </pre>
          </div>
        </div>
      </div>
      <div className="container" style={{paddingTop: '2rem'}}>
        <div className="row">
          <div className="col col--8 col--offset-2">
            <Heading as="h2" className="text--center" style={{marginBottom: '0.5rem'}}>
              Or Use Standalone
            </Heading>
            <p className="text--center" style={{color: 'var(--ifm-color-secondary-darkest)', marginBottom: '1.5rem'}}>
              No framework needed — memharness works with any Python code
            </p>
            <pre style={{
              padding: '1.5rem',
              borderRadius: '8px',
              fontSize: '0.85rem',
              lineHeight: '1.6',
            }}>
              <code>{`from memharness import MemoryHarness
from memharness.agents import ContextAssemblyAgent

async with MemoryHarness("sqlite:///memory.db") as harness:
    # Store memories
    await harness.add_conversational("thread-1", "user", "I prefer Python over JS")
    await harness.add_knowledge("Python 3.13 has free-threaded mode", source="docs")
    await harness.add_entity("Alice", "PERSON", "Senior engineer at Acme Corp")

    # Assemble context for any LLM (BEFORE-loop pattern)
    ctx_agent = ContextAssemblyAgent(harness)
    ctx = await ctx_agent.assemble("Tell me about Python", thread_id="thread-1")

    # Get as LangChain messages (HumanMessage, AIMessage, SystemMessage)
    messages = ctx.to_messages()

    # Or as markdown prompt string
    prompt = ctx.to_prompt()`}</code>
            </pre>
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
      description="Framework-agnostic memory infrastructure for AI agents. 10 memory types, pluggable backends, async-first API.">
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
