import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    'getting-started',
    'usage-langchain',
    {
      type: 'category',
      label: 'Core Concepts',
      items: [
        'concepts/memory-types',
        'concepts/deterministic-vs-ai',
        'concepts/memory-lifecycle',
        'concepts/configuration',
      ],
    },
    {
      type: 'category',
      label: 'Memory Types',
      items: [
        'memory-types/conversational',
        'memory-types/knowledge-base',
        'memory-types/entity',
        'memory-types/workflow',
        'memory-types/toolbox',
        'memory-types/summary',
        'memory-types/tool-log',
        'memory-types/file',
        'memory-types/persona',
      ],
    },
    {
      type: 'category',
      label: 'Backends',
      items: [
        'backends/sqlite',
        'backends/postgresql',
        'backends/in-memory',
      ],
    },
    {
      type: 'category',
      label: 'Embedded Agents',
      items: [
        'agents/overview',
        'agents/summarizer',
        'agents/entity-extractor',
        'agents/consolidator',
        'agents/context-assembler',
      ],
    },
    {
      type: 'category',
      label: 'Memory Tools',
      items: [
        'tools/overview',
        'tools/self-exploration',
        'tools/toolbox-vfs',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/harness',
        'api/types',
        'api/config',
        'api/backends',
        'api/agents',
      ],
    },
  ],
};

export default sidebars;
