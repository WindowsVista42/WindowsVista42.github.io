+++
title = "Lizard Wizard"
date = 2021-07-01

[taxonomies]
tags = [ "engine", "rendering", "ecs" ]
+++

[Repository Link](https://github.com/WindowsVista42/Lizard-Wizard-Parberry-4210)  
A game created by me, Ethan Flute, and Solomon Weatherby for Ian Parberry's 4210 Intro to Game Programming I.

<!-- more -->

## What is it
The premise of Lizard Wizard was that you were a lizard who happened to be a wizard, and you would have to fight through dungeons to get to the next level and eventually beat the game.
At the end of each of these levels you would have some kind of boss that you would have to kill to proceed and the end of the game you would have a larger bigger badder boss to fight.

In short, it was set up to be a 3D roguelike, and we had roughly 3 months to implement everything.

## Rendering
The first problem that had to be tackled for this project was to build a prototype 3D renderer for visualization. Parberry's game engine used
[**DirectX12**](https://docs.microsoft.com/en-us/windows/win32/direct3d12/what-is-directx-12-) under the hood, and more specificially
[**DirectX12TK**](https://github.com/microsoft/DirectXTK12). To build a 3D renderer in his engine, I first had to extend the **LRenderer3D** class and add the neccessary
stuff to get 3D rendering up and running. I won't go into the details of how DirectX12 works here, but for the prototype renderer I used DirectX12TK's
[**Primitive Batch**](https://github.com/microsoft/DirectXTK12/wiki/PrimitiveBatch). This allowed me and my team to push some vertex info to the gpu while I got to work on
a deferred renderer.

The rendering architecture I decided on for was a [**deferred renderer**](https://en.wikipedia.org/wiki/Deferred_shading). The benefit of this kind of approach
is that you get all of the vertex information after all of the vertices have been processed. Because of this, you don't have to worry about any data dependence
and you can process all rendering effects as post process effects. In the case of Lizard Wizard, this allowed me to create some simple lighting,
and my own take on [**ambient occlusion**](https://en.wikipedia.org/wiki/Ambient_occlusion). I chose this kind of architecture because our target platform was
high performance desktop PC's, and the high memory bandwidth usage isn't as much of a problem like it would be on mobile.

## Entity Component System
For this project I built an [**entity component system**](https://en.wikipedia.org/wiki/Entity_component_system) from scratch to help us manage all of the stuff we would have in the scene.

## Procedural Generation

## Movement

## Time Management

## Lessons Learned
